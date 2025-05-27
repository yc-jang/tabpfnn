import json
import pandas as pd
import plotly.express as px
from typing import Dict, Any, Union, List
from functools import reduce


def explode_prediction_dict(pred_dict: Dict[Any, Union[float, List[float]]], name: str) -> pd.DataFrame:
    """
    Converts a dict of index -> scalar or list of values into a normalized DataFrame
    """
    records = []
    for idx, value in pred_dict.items():
        if isinstance(value, list):
            for step, val in enumerate(value):
                records.append({'index': idx, 'step': step, name: val})
        else:
            records.append({'index': idx, 'step': 0, name: value})
    return pd.DataFrame(records)


def is_multistep_dict(pred_dict: Dict[Any, Any]) -> bool:
    """
    Check if the first few items contain list-type values (indicating multistep prediction)
    """
    for val in list(pred_dict.values())[:5]:
        if isinstance(val, list):
            return True
    return False


def plot_predictions_line_safe(
    y_test: Dict[Any, Union[float, List[float]]],
    preds_dict: Dict[str, Dict[Any, Union[float, List[float]]]],
    index: List[Any],
    target: str
):
    """
    Automatically handles both single prediction and multistep prediction formats.
    """
    if is_multistep_dict(y_test):
        # Multistep (index → list of values)
        df_y = explode_prediction_dict(y_test, 'y_true')
        dfs = [df_y]
        for model_name, pred in preds_dict.items():
            df_pred = explode_prediction_dict(pred, model_name)
            dfs.append(df_pred)

        df_merged = reduce(lambda l, r: pd.merge(l, r, on=['index', 'step'], how='outer'), dfs)
        df_long = df_merged.melt(id_vars=['index', 'step'], var_name='model', value_name='value')
        df_long = df_long.sort_values(by=['index', 'step'])

        fig = px.line(
            df_long,
            x='step',
            y='value',
            color='model',
            facet_col='index',
            title=f"Target: {target} - Multistep Prediction"
        )
    else:
        # Single prediction (index → scalar value)
        df = pd.DataFrame({
            'index': index,
            'y_true': [y_test[str(i)] for i in index],
            **{
                model_name: [pred[str(i)] for i in index]
                for model_name, pred in preds_dict.items()
            }
        })

        df_long = df.melt(id_vars='index', var_name='model', value_name='value')
        fig = px.line(
            df_long,
            x='index',
            y='value',
            color='model',
            title=f"Target: {target} - Single Prediction"
        )

    return fig


def plot_from_saved_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    for target, model_result in result.items():
        y_test = model_result['y_test']
        index = model_result['index']

        preds_dict = {
            'tabpfn': model_result['tabpfn'],
            'xgb': model_result['xgb'],
            'ensemble': model_result['ensemble']
        }

        fig = plot_predictions_line_safe(
            y_test=y_test,
            preds_dict=preds_dict,
            index=index,
            target=target
        )

        fig.show()

import json
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from functools import reduce


def stack_flat_values(values: List[float], index_list: List[Any], name: str) -> pd.DataFrame:
    """
    Converts a list of values and corresponding (possibly duplicate) index list
    into a DataFrame with [index, step, value]
    """
    if len(values) != len(index_list):
        raise ValueError(f"Length mismatch: values ({len(values)}), index ({len(index_list)})")

    df = pd.DataFrame({
        'index': index_list,
        name: values
    })

    # Assign step per repeated index
    df['step'] = df.groupby('index').cumcount()
    return df


def plot_predictions_line_safe(
    y_test: List[float],
    preds_dict: Dict[str, List[float]],
    index: List[Any],
    target: str
):
    """
    Robust line plot for list-type predictions, where index may contain repeated entries.
    Produces multi-line subplot per index with step-based x-axis.
    """
    # 1. Convert y_true
    df_y = stack_flat_values(y_test, index, 'y_true')
    dfs = [df_y]

    # 2. Convert model predictions
    for model_name, pred_values in preds_dict.items():
        df_pred = stack_flat_values(pred_values, index, model_name)
        dfs.append(df_pred)

    # 3. Merge all on [index, step]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['index', 'step'], how='outer'), dfs)

    # 4. Melt into long-form for plotly
    df_long = df_merged.melt(id_vars=['index', 'step'], var_name='model', value_name='value')
    df_long = df_long.sort_values(by=['index', 'step'])

    # 5. Plot
    fig = px.line(
        df_long,
        x='step',
        y='value',
        color='model',
        facet_col='index',
        title=f"Target: {target} - Multi-step Prediction by Index"
    )

    return fig


def plot_from_saved_json(file_path: str):
    """
    Loads JSON file with saved prediction results and calls plot_predictions_line_safe
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    for target, model_result in result.items():
        y_test = model_result['y_test']
        index = model_result['index']

        preds_dict = {
            'tabpfn': model_result['tabpfn'],
            'xgb': model_result['xgb'],
            'ensemble': model_result['ensemble']
        }

        fig = plot_predictions_line_safe(
            y_test=y_test,
            preds_dict=preds_dict,
            index=index,
            target=target
        )

        fig.show()
