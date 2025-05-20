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


def interpret_prediction_change_simple(
    model_runner,
    user_input_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    previous_prediction: Optional[np.ndarray] = None,
    shap_explain: bool = True,
    diff_threshold: float = 1e-5
):
    """
    예측값 해석 (민감도 분석 제거) + SHAP 기반 설명 제공
    """
    pred = model_runner.predict(predicted_df)
    display(HTML(f"<h4>예측값: {pred.round(5).tolist()}</h4>"))

    if previous_prediction is not None:
        change = np.abs(pred - previous_prediction)
        if np.all(change < diff_threshold):
            display(HTML("<div style='color:red; font-weight:bold;'>※ 입력을 변경하였지만 예측값이 동일하거나 거의 변화하지 않았습니다.</div>"))
            display(HTML("""
            <p style="font-size:14px;">
            이는 다음과 같은 이유로 발생할 수 있습니다:
            <ul style="font-size:13px;">
              <li>해당 입력값이 모델 내부의 트리 구조에서 같은 분기 경로를 따를 경우, 결과는 동일하게 유지됩니다.</li>
              <li>SHAP에서 높은 영향도를 가진 변수라 하더라도, 현재 입력 조합에서는 영향력이 미미할 수 있습니다.</li>
              <li>모델이 변수 간 상호작용(Interaction)을 고려하기 때문에, 단일 변수 변화로는 예측에 영향을 주지 않을 수 있습니다.</li>
              <li>즉, 전체 모델 학습 구조 상 특정 그룹(leaf)에 속하게 되어 평균 예측값이 고정됩니다.</li>
            </ul>
            </p>
            """))

            aligned_input = predicted_df[model_runner.model.feature_names_in_]
            explainer = shap.Explainer(model_runner.model)
            shap_values = explainer(aligned_input)

            if shap_explain:
                display(HTML("<b>SHAP 영향도 분석 (Waterfall Plot)</b>"))
                shap.plots.waterfall(shap_values[0])

            display(HTML("""
            <p style="font-size:13px; color:gray;">
            ※ SHAP 그래프는 현재 예측에서 어떤 변수가 상대적으로 영향을 많이 주었는지를 보여줍니다.<br>
            다만 영향도가 높게 나타난다고 해서, 해당 변수를 바꿨을 때 항상 예측이 바뀐다는 보장은 없습니다.<br>
            이는 모델이 다수의 변수 조합을 기반으로 판단하기 때문입니다.
            </p>
            """))

    return pred

def interpret_prediction_change_full(
    model_runner,
    user_input_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    previous_prediction: Optional[np.ndarray] = None,
    shap_explain: bool = True,
    diff_threshold: float = 1e-5
):
    """
    예측 결과 해석: 변화 감지 + 사용자 친화적 설명 + SHAP 분석 + 트리 구조 근거 포함
    """
    import shap
    from IPython.display import display, HTML
    import pandas as pd

    pred = model_runner.predict(predicted_df)
    display(HTML(f"<h4>예측값: {pred.round(5).tolist()}</h4>"))

    if previous_prediction is not None:
        change = np.abs(pred - previous_prediction)
        if np.all(change < diff_threshold):
            display(HTML("<div style='color:red; font-weight:bold;'>※ 입력을 변경하였지만 예측값이 동일하거나 거의 변화하지 않았습니다.</div>"))
            display(HTML("""
            <p style="font-size:14px;">
            입력값 변화에도 예측이 고정되는 이유는 다음과 같습니다:
            <ul style="font-size:13px;">
              <li><b>트리 구조의 분기 조건</b>에 따라 현재 입력이 동일한 리프 노드에 도달할 수 있습니다.</li>
              <li><b>SHAP에서 중요도가 높아도</b>, 현재 입력 조건에서는 영향력이 0에 가까울 수 있습니다.</li>
              <li><b>변수 간 상호작용</b>이 중요한 경우, 단일 변수 변화만으로는 모델 반응이 없습니다.</li>
            </ul>
            아래에 해당 예측에서 실제 모델이 어떻게 반응했는지를 정량적으로 제시합니다.
            </p>
            """))

            # 1. SHAP 분석
            aligned_input = predicted_df[model_runner.model.feature_names_in_]
            explainer = shap.Explainer(model_runner.model)
            shap_values = explainer(aligned_input)

            if shap_explain:
                display(HTML("<b>1. SHAP 영향도 분석 (Waterfall Plot)</b>"))
                shap.plots.waterfall(shap_values[0])

            # 2. 모델의 트리 split 빈도 확인
            booster = model_runner.model.get_booster()
            split_importance = booster.get_score(importance_type='weight')
            split_df = pd.DataFrame.from_dict(split_importance, orient='index', columns=['split_count']).sort_values(by='split_count', ascending=False)
            display(HTML("<b>2. 모델이 자주 분기한 변수 (split count 기준)</b>"))
            display(split_df.head(10).style.format({'split_count': '{:.0f}'}))

            # 3. 해당 변수의 트리 split 경로 확인
            shap_df = pd.DataFrame(shap_values.values, columns=aligned_input.columns)
            top_feature = shap_df.abs().mean().sort_values(ascending=False).index[0]

            tree_df = booster.trees_to_dataframe()
            feature_splits = tree_df[tree_df['Feature'] == top_feature]

            if feature_splits.empty:
                display(HTML(f"<b>3. '{top_feature}' 변수는 트리 split에 사용되지 않았습니다.</b>"))
            else:
                display(HTML(f"<b>3. '{top_feature}' 변수의 분기 조건 예시</b>"))
                display(feature_splits[['Tree', 'Node', 'Split', 'Yes', 'No', 'Missing']].head(5))

            display(HTML(f"""
            <p style="font-size:13px; color:gray;">
            ※ 위 split 조건은 모델이 '{top_feature}' 변수를 어떤 조건에서 활용했는지를 보여줍니다.<br>
            그러나 현재 입력이 이 조건을 만족하지 않으면 예측값에는 영향이 없을 수 있습니다.
            </p>
            """))

    return pred


def interpret_prediction_change_full(
    model_runner,
    user_input_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    previous_prediction: Optional[np.ndarray] = None,
    shap_explain: bool = True,
    diff_threshold: float = 1e-5
):
    """
    예측 결과 해석: 변화 감지 + 사용자 친화적 설명 + SHAP 분석 + 트리 구조 기반 설명 + 조정 가이드
    """
    import shap
    from IPython.display import display, HTML
    import pandas as pd

    pred = model_runner.predict(predicted_df)
    display(HTML(f"<h4>예측값: {pred.round(5).tolist()}</h4>"))

    if previous_prediction is not None:
        change = np.abs(pred - previous_prediction)
        if np.all(change < diff_threshold):
            display(HTML("<div style='color:red; font-weight:bold;'>※ 입력을 변경하였지만 예측값이 동일하거나 거의 변화하지 않았습니다.</div>"))
            display(HTML("""
            <p style="font-size:14px;">
            입력값 변화에도 예측이 고정되는 이유는 다음과 같습니다:
            <ul style="font-size:13px;">
              <li><b>트리 구조의 분기 조건</b>에 따라 현재 입력이 동일한 리프 노드에 도달할 수 있습니다.</li>
              <li><b>SHAP에서 중요도가 높아도</b>, 현재 입력 조건에서는 영향력이 0에 가까울 수 있습니다.</li>
              <li><b>변수 간 상호작용</b>이 중요한 경우, 단일 변수 변화만으로는 모델 반응이 없습니다.</li>
            </ul>
            아래에 해당 예측에서 실제 모델이 어떻게 반응했는지를 정량적으로 제시합니다.
            </p>
            """))

            # 1. SHAP 분석
            aligned_input = predicted_df[model_runner.model.feature_names_in_]
            explainer = shap.Explainer(model_runner.model)
            shap_values = explainer(aligned_input)

            if shap_explain:
                display(HTML("<b>1. SHAP 영향도 분석 (Waterfall Plot)</b>"))
                shap.plots.waterfall(shap_values[0])

            # 2. 모델의 트리 split 빈도 확인
            booster = model_runner.model.get_booster()
            split_importance = booster.get_score(importance_type='weight')
            split_df = pd.DataFrame.from_dict(split_importance, orient='index', columns=['split_count']).sort_values(by='split_count', ascending=False)
            display(HTML("<b>2. 모델이 자주 분기한 변수 (split count 기준)</b>"))
            display(split_df.head(10).style.format({'split_count': '{:.0f}'}))

            # 3. 상위 SHAP 변수 3개에 대해 분기 조건 + 가이드 제공
            shap_df = pd.DataFrame(shap_values.values, columns=aligned_input.columns)
            top_features = shap_df.abs().mean().sort_values(ascending=False).head(3).index

            tree_df = booster.trees_to_dataframe()

            for top_feature in top_features:
                feature_splits = tree_df[tree_df['Feature'] == top_feature]
                display(HTML(f"<b>3. '{top_feature}' 변수의 분기 조건 예시</b>"))

                if feature_splits.empty:
                    display(HTML(f"<i>{top_feature}는 트리 분기에 사용되지 않았습니다.</i>"))
                    continue

                display(feature_splits[['Tree', 'Node', 'Split', 'Yes', 'No', 'Missing']].head(5))

                suggested = feature_splits['Split'].dropna()
                current_val = user_input_df[top_feature].iloc[0]

                if suggested.empty:
                    continue

                if (suggested < current_val).all():
                    direction = "낮춰보세요"
                elif (suggested > current_val).all():
                    direction = "높여보세요"
                else:
                    direction = "적절한 범위 내에서 시도해보세요"

                target_range = f"{suggested.min():.2f} ~ {suggested.max():.2f}"

                display(HTML(f"""
                <p style="font-size:13px; color:blue;">
                현재 입력값은 <b>{top_feature} = {current_val:.2f}</b> 입니다.<br>
                이 변수의 분기 조건을 고려할 때 예측 반응을 얻기 위해서는 값을 <b>{direction}</b>.<br>
                권장 조정 범위: <b>{target_range}</b>
                </p>
                """))

    return pred

