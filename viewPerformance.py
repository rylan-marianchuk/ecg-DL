import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc


def AUC(y_true, y_score):
    """

    :param y_true: binary
    :param y_score: sigmoid probabilities
    :return:
    """
    fig_hist = px.histogram(
        x=y_score, color=y_true, nbins=70,
        labels=dict(color='True Labels', x='Score')
    )
    fig_hist.show()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()
    return


def regression_performance(y_true, y_pred, hovertexts, ds):
    df = pd.DataFrame()
    df['prediction'] = y_pred
    df['true'] = y_true
    df['l'] = [str(ds.filenames.index(s.split("-")[0])) + " " + s.split("-")[-1] for s in hovertexts]

    fig = px.scatter(
        df, x='true', y='prediction',
        hover_data='l',
        marginal_x='histogram', marginal_y='histogram'
    )
    fig.update_traces(histnorm='probability', selector={'type': 'histogram'}, marker=dict(color='#4C8577'))
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_true.min(), y0=y_true.min(),
        x1=y_true.max(), y1=y_true.max()
    )

    fig.show()



