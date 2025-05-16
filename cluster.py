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

    def run_kmeans(self, cols: list[str], k: int) -> KMeans:
        """지정된 컬럼 기준으로 KMeans 클러스터링 수행"""
        km = KMeans(n_clusters=k, random_state=self.random_seed)
        km.fit(self.df[cols])
        self.kmeans_model = km
        return km

    def make_representatives(self, cols: list[str], km: KMeans) -> None:
        """각 클러스터의 대표값(평균)을 저장"""
        self.df['cluster'] = km.labels_
        for c in range(km.n_clusters):
            self.cluster_values[c] = (
                self.df[self.df['cluster'] == c]
                .drop(columns='cluster')
                .mean(numeric_only=True)
            )

    def predict_other_cols(self, km: KMeans, user_input: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """유저 입력으로부터 클러스터 예측 후, 해당 클러스터의 대표값 기반 전체 feature 생성"""
        if isinstance(user_input, pd.Series):
            user_input = user_input.to_frame().T

        cluster_labels = km.predict(user_input)
        results = []

        for i, cluster in enumerate(cluster_labels):
            base = self.cluster_values[cluster].copy()
            base.update(user_input.iloc[i])
            results.append(base)

        return pd.DataFrame(results).reset_index(drop=True)

    def process(self, user_control_columns: list[str], user_input: Union[pd.Series, pd.DataFrame], k: int = 5) -> pd.DataFrame:
        """전체 프로세스 실행: 클러스터링 후 입력값 기반 전체 feature 생성"""
        km = self.run_kmeans(user_control_columns, k)
        self.make_representatives(user_control_columns, km)
        return self.predict_other_cols(km, user_input)

    def summarize_clusters(self) -> pd.DataFrame:
        """클러스터별 샘플 수와 주요 통계 요약"""
        if 'cluster' not in self.df.columns:
            raise ValueError("클러스터 정보를 먼저 생성해야 합니다. run_kmeans 이후 make_representatives를 호출하세요.")

        summary = self.df.groupby('cluster').agg(['count', 'mean', 'std'])
        return summary

    def find_optimal_k(self, cols: list[str], k_range: range = range(2, 11)) -> int:
        """최적의 클러스터 수 찾기 (Silhouette Score 기준, Plotly 시각화 포함)"""
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
