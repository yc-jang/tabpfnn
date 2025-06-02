import pandas as pd
import re

class LagFeatureGenerator:
    def __init__(self, lot_col='LOT_ID', target_col='discharge',
                 n_lags=3, mark_missing_value=-999):
        self.lot_col = lot_col
        self.target_col = target_col
        self.n_lags = n_lags
        self.mark_missing_value = mark_missing_value
        self.group_stats_ = None

    def _parse_lot(self, lot_id):
        match = re.match(r'(.{9})(\d{6})(\d{2})$', lot_id)
        if not match:
            return pd.Series({'LOT_group': None, 'LOT_line': None,
                              'LOT_date': None, 'LOT_serial': None})
        return pd.Series({
            'LOT_group': match.group(1) + match.group(2),
            'LOT_line': lot_id[8],
            'LOT_date': match.group(2),
            'LOT_serial': int(match.group(3)),
        })

    def _ensure_lot_info(self, df):
        if not {'LOT_group', 'LOT_line', 'LOT_date', 'LOT_serial'}.issubset(df.columns):
            parsed = df[self.lot_col].apply(self._parse_lot)
            df = pd.concat([df, parsed], axis=1)
        return df

    def fit(self, df):
        df = df.copy()
        df = self._ensure_lot_info(df)
        df.sort_values(['LOT_line', 'LOT_date', 'LOT_serial'], inplace=True)

        # lag 피처 생성용 데이터 저장
        self.group_stats_ = (
            df[['LOT_group', self.target_col]]
            .drop_duplicates(subset='LOT_group')
            .reset_index(drop=True)
        )
        return self

    def transform(self, df):
        df = df.copy()
        df = self._ensure_lot_info(df)

        # merge용 기준 (group -> lag값)
        df = df.merge(
            self.group_stats_,
            on='LOT_group',
            how='left',
            suffixes=('', '_current')
        )

        # 순서 유지 위해 정렬
        self.group_stats_.sort_values(by='LOT_group', inplace=True)
        group_list = self.group_stats_['LOT_group'].tolist()
        discharge_list = self.group_stats_[self.target_col].tolist()

        # lag dictionary 생성
        lag_df = pd.DataFrame({ 'LOT_group': group_list })
        for i in range(1, self.n_lags + 1):
            lag_col = f'{self.target_col}_lag_{i}'
            lag_df[lag_col] = [self.mark_missing_value]*i + discharge_list[:-i]

        # merge with lag info
        df = df.merge(lag_df, on='LOT_group', how='left')
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
