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
