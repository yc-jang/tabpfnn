# Re-defining the function after kernel reset
import plotly.graph_objects as go
import numpy as np

def plot_predictions_with_insights(y_true, predictions_dict, title="Prediction vs Actual with Insights"):
    fig = go.Figure()
    insights = []

    # 각 모델의 예측과 성능 지표 시각화
    for model_name, result in predictions_dict.items():
        y_pred = np.array(result['y_pred'])
        r2 = result['r2']
        rmse = result['rmse']
        residuals = y_true - y_pred

        # 산점도 추가
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=y_true,
            mode='markers',
            name=f"{model_name} (R²={r2:.2f}, RMSE={rmse:.2f})",
            opacity=0.7
        ))

        # 인사이트 요약용
        insights.append({
            'model': model_name,
            'r2': r2,
            'rmse': rmse,
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'over_pred_ratio': np.mean(residuals < 0),
            'under_pred_ratio': np.mean(residuals > 0)
        })

    # 완벽한 예측선
    min_val = min(np.min(y_true), *(np.min(np.array(v['y_pred'])) for v in predictions_dict.values()))
    max_val = max(np.max(y_true), *(np.max(np.array(v['y_pred'])) for v in predictions_dict.values()))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideal',
        line=dict(color='black', dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        legend=dict(x=0.02, y=0.98),
        height=650
    )

    # 인사이트 텍스트 출력
    insight_text = "### Prediction Insight Summary\n"
    for i in insights:
        insight_text += (
            f"- **{i['model']}**: R² = {i['r2']:.3f}, RMSE = {i['rmse']:.3f}, "
            f"Residual Mean = {i['residual_mean']:.4f}, Std = {i['residual_std']:.4f}, "
            f"Over-pred: {i['over_pred_ratio']:.1%}, Under-pred: {i['under_pred_ratio']:.1%}\n"
        )

    return fig, insight_text

import plotly.graph_objects as go

def plot_prediction_lines_with_lot(lot_index, y_true, predictions_dict, title="Prediction Comparison by LOT"):
    """
    x축: LOT(샘플 순서), y축: 실제 및 예측값 선 그래프
    """
    fig = go.Figure()

    # 실제값 라인
    fig.add_trace(go.Scatter(
        x=lot_index,
        y=y_true,
        mode='lines+markers',
        name="Actual",
        line=dict(color='black', width=2, dash='solid')
    ))

    # 예측 모델별 라인
    for model_name, result in predictions_dict.items():
        fig.add_trace(go.Scatter(
            x=lot_index,
            y=result['y_pred'],
            mode='lines+markers',
            name=model_name,
            opacity=0.85
        ))

    fig.update_layout(
        title=title,
        xaxis_title="LOT Index",
        yaxis_title="Value",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
        height=600
    )

    return fig


