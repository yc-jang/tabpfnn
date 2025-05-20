import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from typing import Union


class Clustering:
    def __init__(self, data: pd.DataFrame, random_seed: int = 42):
        self.df = data.copy()
        self.random_seed = random_seed
        self.cluster_values = {}
        self.kmeans_model = None
        self.user_control_columns = []

    def run_kmeans(self, cols: list[str], k: int) -> KMeans:
        """KMeans 클러스터링 수행"""
        self.user_control_columns = cols
        km = KMeans(n_clusters=k, random_state=self.random_seed)
        km.fit(self.df[cols])
        self.kmeans_model = km
        self.df['cluster'] = km.labels_
        return km

    def make_representatives(self) -> None:
        """각 클러스터의 비제어 컬럼 대표값 생성"""
        if 'cluster' not in self.df.columns:
            raise ValueError("run_kmeans()를 먼저 실행해야 합니다.")
        non_control_cols = [col for col in self.df.columns if col not in self.user_control_columns + ['cluster']]
        for c in range(self.kmeans_model.n_clusters):
            self.cluster_values[c] = (
                self.df[self.df['cluster'] == c][non_control_cols]
                .mean(numeric_only=True)
            )

    def predict_other_cols(self, user_input: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """user_control_columns 기준 클러스터 예측 및 비제어 컬럼 채움"""
        if self.kmeans_model is None or not self.cluster_values:
            raise ValueError("클러스터링이 먼저 수행되어야 합니다.")
        if isinstance(user_input, pd.Series):
            user_input = user_input.to_frame().T

        cluster_labels = self.kmeans_model.predict(user_input[self.user_control_columns])
        results = []

        for i, cluster in enumerate(cluster_labels):
            base = self.cluster_values[cluster].copy()
            base.update(user_input.iloc[i])
            results.append(base)

        return pd.DataFrame(results).reset_index(drop=True)

    def summarize_clusters(self) -> pd.DataFrame:
        """클러스터별 샘플 수 및 요약 통계"""
        if 'cluster' not in self.df.columns:
            raise ValueError("클러스터 정보 없음. run_kmeans() 이후 실행하세요.")
        return self.df.groupby('cluster').agg(['count', 'mean', 'std'])

    def find_optimal_k(self, cols: list[str], k_range: range = range(2, 11)) -> int:
        """최적 클러스터 수 자동 탐색 (Silhouette 기준) 및 Plotly 시각화"""
        scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_seed)
            labels = km.fit_predict(self.df[cols])
            score = silhouette_score(self.df[cols], labels)
            scores.append((k, score))

        ks, sil_scores = zip(*scores)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=sil_scores, mode='lines+markers', name='Silhouette Score'))
        fig.update_layout(
            title="클러스터 수에 따른 Silhouette Score",
            xaxis_title="클러스터 수 (k)",
            yaxis_title="Silhouette Score",
            template="plotly_white",
            height=400
        )
        fig.show()

        best_k = max(scores, key=lambda x: x[1])[0]
        print(f"[INFO] 최적 k: {best_k} (Silhouette Score 기준)")
        return best_k

    def process(
        self,
        user_control_columns: list[str],
        user_input: Union[pd.Series, pd.DataFrame],
        k: int = None,
        k_range: range = range(2, 11)
    ) -> pd.DataFrame:
        """
        전체 프로세스: 최적 k 탐색 (옵션) → 클러스터링 → 대표값 생성 → 유저 입력으로 전체 row 생성
        """
        if k is None:
            k = self.find_optimal_k(user_control_columns, k_range)
        self.run_kmeans(user_control_columns, k)
        self.make_representatives()
        return self.predict_other_cols(user_input)


import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

class UserControlInputWidget:
    def __init__(self, user_control_columns: list[str], callback_function=None):
        """
        user_control_columns: 입력 받을 컬럼 리스트
        callback_function: 입력 완료 시 실행할 함수 (예: Clustering.process)
        """
        self.user_control_columns = user_control_columns
        self.callback_function = callback_function
        self.widgets_dict = {
            col: widgets.FloatText(description=col, layout=widgets.Layout(width='200px'))
            for col in user_control_columns
        }
        self.submit_button = widgets.Button(description="입력 완료", button_style='success')
        self.output = widgets.Output()
        self.user_input_df = None
        self._build_ui()

    def _build_ui(self):
        widget_list = list(self.widgets_dict.values())
        rows = [widgets.HBox(widget_list[i:i+3]) for i in range(0, len(widget_list), 3)]
        form = widgets.VBox(rows + [self.submit_button, self.output])
        display(form)
        self.submit_button.on_click(self._on_submit)

    def _on_submit(self, b):
        with self.output:
            clear_output()
            values = {col: widget.value for col, widget in self.widgets_dict.items()}
            self.user_input_df = pd.DataFrame([values])
            display(widgets.HTML("<b>입력값 확인:</b>"))
            display(self.user_input_df)

            if self.callback_function:
                result = self.callback_function(self.user_input_df)
                display(widgets.HTML("<b>클러스터 기반 예측 결과:</b>"))
                display(result)

    def get_user_input(self) -> pd.DataFrame:
        return self.user_input_df

import pandas as pd
import numpy as np
from typing import Optional
from IPython.display import display, HTML
import shap
import plotly.graph_objects as go

def interpret_prediction_change(
    model_runner,
    user_input_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    previous_prediction: Optional[np.ndarray] = None,
    sensitivity_feature: Optional[str] = None,
    sensitivity_values: Optional[list] = None,
    shap_explain: bool = True,
    diff_threshold: float = 1e-5
):
    """
    예측값을 기반으로 입력 변화에 대한 반응을 해석하고,
    변화가 없을 경우 사용자에게 그 이유와 인사이트를 제공하는 함수.
    """
    # 1. 예측값 계산
    pred = model_runner.predict(predicted_df)
    display(HTML(f"<h4>예측값: {pred.round(5).tolist()}</h4>"))

    # 2. 예측값 변화 여부 확인
    if previous_prediction is not None:
        change = np.abs(pred - previous_prediction)
        if np.all(change < diff_threshold):
            display(HTML("<div style='color:red; font-weight:bold;'>※ 입력을 변경하였지만 예측값이 동일하거나 거의 변화하지 않았습니다.</div>"))
            display(HTML("""
            <p style="font-size:14px;">
            입력하신 값이 AI가 학습한 판단 경로에서는 영향을 거의 주지 않는다고 해석됩니다.<br>
            이는 모델이 해당 입력을 이미 알고 있는 특정 그룹으로 분류하고, 해당 그룹의 평균적인 결과를 반환하기 때문입니다.
            </p>
            """))

            # 3. SHAP 영향도 분석
            aligned_input = predicted_df[model_runner.model.feature_names_in_]
            explainer = shap.Explainer(model_runner.model)
            shap_values = explainer(aligned_input)

            if shap_explain:
                display(HTML("<b>SHAP 영향도 분석 (Waterfall Plot)</b>"))
                shap.plots.waterfall(shap_values[0])

            # 4. SHAP 기준 상위 feature 민감도 분석
            shap_df = pd.DataFrame(shap_values.values, columns=aligned_input.columns)
            shap_mean = shap_df.abs().mean().sort_values(ascending=False)
            top_feature = sensitivity_feature or shap_mean.index[0]

            display(HTML(f"<b>주요 변수 '{top_feature}'에 대한 민감도 분석</b>"))

            if sensitivity_values is None:
                base_val = user_input_df[top_feature].iloc[0]
                sensitivity_values = [base_val * ratio for ratio in [0.8, 0.9, 1.0, 1.1, 1.2]]

            result_rows = []
            for val in sensitivity_values:
                modified = user_input_df.copy()
                modified[top_feature] = val

                full_input = []
                for _, row in modified.iterrows():
                    d = {}
                    for col in model_runner.model.feature_names_in_:
                        if col in row:
                            d[col] = row[col]
                        else:
                            d[col] = reference_df[col].mean() if col in reference_df else 0
                    full_input.append(d)

                full_df = pd.DataFrame(full_input)
                p = model_runner.predict(full_df)[0]
                result_rows.append((val, p))

            # 5. 민감도 그래프
            values, preds = zip(*result_rows)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=values, y=preds, mode='lines+markers', name='Prediction'))
            fig.update_layout(
                title=f"'{top_feature}' 변수 변화에 따른 예측값 민감도",
                xaxis_title=top_feature,
                yaxis_title="예측값",
                template="plotly_white",
                height=400
            )
            fig.show()

            display(HTML("""
            <p style="font-size:13px; color:gray;">
            ※ 위 그래프는 주요 변수의 값이 바뀔 때 예측이 어떻게 달라지는지를 보여줍니다.<br>
            예측값 변화가 거의 없다면, 해당 변수는 현재 입력 조합에서는 영향력이 낮은 것입니다.
            </p>
            """))

    return pred
