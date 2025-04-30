
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import shap
import plotly.graph_objects as go
from typing import Optional

# TabPFN imports
from tabpfn import TabPFNRegressor
from tabpfn_extensions.interpretability import get_shap_values, plot_shap

TABPFN_CKPT_PATH = 'models/pfn/tabpfn-v2-regressor.ckpt'


class TabPFNRunner:
    def __init__(self, checkpoint_path: str = TABPFN_CKPT_PATH, device: str = 'cpu'):
        self.model = TabPFNRegressor(checkpoint_path=checkpoint_path, device=device)
        self.trained = False
        self.shap_values_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'rmse': mean_squared_error(y, y_pred, squared=False),
            'y_pred': y_pred
        }

    def explain(self, X, attribute_names=None):
        if not self.trained:
            raise ValueError("Model not trained yet.")
        self.shap_values_ = get_shap_values(self.model, X, attribute_names=attribute_names)
        return self.shap_values_

    def plot_shap(self):
        if self.shap_values_ is not None:
            plot_shap(self.shap_values_)
        else:
            print("No SHAP values found. Run explain() first.")


class XGBoostRunner:
    def __init__(self):
        self.model = XGBRegressor()
        self.explainer = None
        self.shap_values_ = None
        self.trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.explainer = shap.Explainer(self.model)
        self.trained = True

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'rmse': mean_squared_error(y, y_pred, squared=False),
            'y_pred': y_pred
        }

    def explain(self, X):
        if not self.trained:
            raise ValueError("Model must be trained before explanation.")
        self.shap_values_ = self.explainer(X)
        return self.shap_values_

    def plot_shap_summary(self):
        if self.shap_values_ is not None:
            shap.plots.beeswarm(self.shap_values_)
        else:
            print("No SHAP values found. Run explain() first.")


class EnsembleRunner:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def predict(self, X):
        y1 = self.model1.predict(X)
        y2 = self.model2.predict(X)
        return (y1 + y2) / 2

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'rmse': mean_squared_error(y, y_pred, squared=False),
            'y_pred': y_pred
        }


def plot_predictions(y_true, predictions_dict):
    fig = go.Figure()
    for label, result in predictions_dict.items():
        fig.add_trace(go.Scatter(
            x=result['y_pred'], y=y_true,
            mode='markers',
            name=f"{label} (R²={result['r2']:.2f}, RMSE={result['rmse']:.2f})",
            opacity=0.7
        ))
    fig.add_trace(go.Scatter(
        x=[y_true.min(), y_true.max()],
        y=[y_true.min(), y_true.max()],
        mode='lines',
        name='Ideal',
        line=dict(color='black', dash='dash')
    ))
    fig.update_layout(
        title="Prediction vs Actual",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=600
    )
    return fig


def compare_shap_summary(tabpfn_shap_df, xgb_shap_df, top_n=10):
    merged = pd.DataFrame({
        'TabPFN': tabpfn_shap_df.abs().mean(axis=0),
        'XGBoost': xgb_shap_df.abs().mean(axis=0)
    })
    merged['feature'] = merged.index
    merged = merged.sort_values(by='TabPFN', ascending=False).head(top_n).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=merged['feature'], y=merged['TabPFN'], name='TabPFN SHAP'))
    fig.add_trace(go.Bar(x=merged['feature'], y=merged['XGBoost'], name='XGBoost SHAP'))

    fig.update_layout(
        title='Feature Importance Comparison (SHAP)',
        barmode='group',
        xaxis_title='Feature',
        yaxis_title='Mean |SHAP value|',
        height=500
    )
    fig.show()
