# Re-import required packages after code execution state reset
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.graph_objects as go

# Fallback for testing without tabpfn_extensions installed
AutoTabPFNRegressor = None
get_shap_values = None
plot_shap = None


class TabPFNPipeline:
    def __init__(self, device='cpu'):
        self.tabpfn_model = AutoTabPFNRegressor(device=device) if AutoTabPFNRegressor else None
        self.xgb_model = XGBRegressor()
        self.trained = False
        self.X_train = None
        self.y_train = None
        self.result = {}

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        if self.tabpfn_model:
            self.tabpfn_model.fit(X_train, y_train)

        self.xgb_model.fit(X_train, y_train)
        self.trained = True

    def evaluate(self, X_test, y_test):
        if not self.trained:
            raise ValueError("Model must be fitted before evaluation.")

        y_pred_tabpfn = self.tabpfn_model.predict(X_test) if self.tabpfn_model else np.zeros(len(y_test))
        y_pred_xgb = self.xgb_model.predict(X_test)
        y_pred_ensemble = (y_pred_tabpfn + y_pred_xgb) / 2

        self.result = {
            'TabPFN': {'y_pred': y_pred_tabpfn, 'r2': r2_score(y_test, y_pred_tabpfn), 'rmse': mean_squared_error(y_test, y_pred_tabpfn, squared=False)},
            'XGBoost': {'y_pred': y_pred_xgb, 'r2': r2_score(y_test, y_pred_xgb), 'rmse': mean_squared_error(y_test, y_pred_xgb, squared=False)},
            'Ensemble': {'y_pred': y_pred_ensemble, 'r2': r2_score(y_test, y_pred_ensemble), 'rmse': mean_squared_error(y_test, y_pred_ensemble, squared=False)},
        }

    def plot_predictions(self, y_test):
        if not self.result:
            raise ValueError("No evaluation results to plot.")

        fig = go.Figure()
        for model, res in self.result.items():
            fig.add_trace(go.Scatter(
                x=res['y_pred'], y=y_test,
                mode='markers',
                name=f"{model} (R²={res['r2']:.2f}, RMSE={res['rmse']:.2f})",
                opacity=0.7
            ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Ideal',
            line=dict(color='black', dash='dash')
        ))
        fig.update_layout(
            title="Predicted vs Actual Values",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            legend_title="Model",
            height=600
        )
        return fig


class TabPFNEnsembleInterpreter(TabPFNPipeline):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.shap_values_ = None
        self.feature_names_ = None

    def explain(self, X_sample):
        if not self.tabpfn_model or not get_shap_values:
            raise ImportError("SHAP explanation modules not available.")
        if not self.trained:
            raise ValueError("Model must be trained before explanation.")

        self.feature_names_ = X_sample.columns if isinstance(X_sample, pd.DataFrame) else [f"feat_{i}" for i in range(X_sample.shape[1])]
        self.shap_values_ = get_shap_values(self.tabpfn_model, X_sample, attribute_names=self.feature_names_)
        return self.shap_values_

    def plot_shap(self):
        if self.shap_values_ is not None and plot_shap:
            plot_shap(self.shap_values_)
        else:
            print("SHAP values not available. Run 'explain()' first.")
