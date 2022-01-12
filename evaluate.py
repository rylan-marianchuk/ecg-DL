import torch
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc


def evaluateContinuous(te_dataloader, ds_test, model, device, batch_size, saveToDisk=False):
    """
    Execute the test set on a trained model, evaluate its predictive performance
    :param te_dataloader: (DataLoader object from torch.utils.data.dataloader) test set dataloader
    :param ds_test (ecgDataset object) test set dataset
    :param model: trained model, callable given input features
    :param device: cpu or cuda
    :param batch_size: (int)
    :param saveToDisk: (bool) whether to save as html file
    """
    model.eval()
    with torch.no_grad():
        y_pred = torch.zeros(ds_test.n_obs)
        y_true = torch.zeros(ds_test.n_obs)
        hovertexts = []
        batch = 0
        for X, y, ids in te_dataloader:
            X = X.to(device)
            y = y.to(device)
            prediction = model(X)
            if ds_test.single_lead_obs:
                l = batch*batch_size*ds_test.n_leads
                r = (batch+1)*batch_size*ds_test.n_leads
            else:
                l = batch*batch_size
                r = (batch+1)*batch_size
            y_pred[l:r] = prediction.flatten()
            y_true[l:r] = y
            hovertexts += ids
            batch += 1

        rmeanloss = torch.mean(torch.sqrt((y_true - y_pred) * (y_true - y_pred)))
        print("Test Root Mean loss:  " + str(rmeanloss))
        regression_performance(y_true, y_pred, hovertexts, saveToDisk)
        bland_altman(y_true, y_pred, hovertexts, saveToDisk)

    if saveToDisk:
        D = {'ids':hovertexts, 'y_true':y_true, 'y_pred':y_pred}
        torch.save(D, "true-pred-test.pt")

def evaluateBinary():
    return


def AUC(y_true, y_score, saveToDisk=False):
    """
    Generate a Receiver Operating Characteristic Curve, indicating its integral
    :param y_true: binary
    :param y_score: sigmoid probabilities
    :param saveToDisk: (bool) whether to save as html file
    """
    fig_hist = px.histogram(
        x=y_score, color=y_true, nbins=70,
        labels=dict(color='True Labels', x='Score')
    )
    if saveToDisk:
        fig_hist.write_html("./Prediction-Histogram.html")
    else:
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

    if saveToDisk:
        fig.write_html("./ROC-AUC.html")
    else:
        fig.show()


def regression_performance(y_true, y_pred, hovertexts, saveToDisk=False):
    """
    Generate a plot of True vs. Predicted Values
    :param y_true: torch tensor shape=(N,)
    :param y_pred: torch tensor shape=(N,)
    :param hovertexts: (list) of identification strings
    :param saveToDisk: (bool) whether to save as html file
    """
    df = pd.DataFrame()
    df['Prediction'] = y_pred
    df['True'] = y_true
    df['l'] = hovertexts

    fig = px.scatter(
        df, y='True', x='Prediction',
        hover_data='l',
        marginal_x='histogram', marginal_y='histogram',
        title="Correlation Plot of True vs. Predicted",
        opacity=0.6
    )

    fig.update_traces(marker=dict(color='#ef8354'))
    fig.update_traces(histnorm='probability', selector={'type': 'histogram'}, marker=dict(color='#536b78'))

    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_true.min(), y0=y_true.min(),
        x1=y_true.max(), y1=y_true.max()
    )

    if saveToDisk:
        fig.write_html("./correlation-true-pred.html")
    else:
        fig.show()


def bland_altman(y_true, y_pred, hovertexts, saveToDisk=False):
    """
    Generate a Bland Altman Plot
    :param y_true: torch tensor shape=(N,)
    :param y_pred: torch tensor shape=(N,)
    :param hovertexts: (list) of identification strings
    :param saveToDisk: (bool) whether to save as html file
    """
    df = pd.DataFrame()
    df["Average:  (y_true + y_pred) / 2"] = (y_true + y_pred) / 2
    df["Difference:  y_true - y_pred"] = y_true - y_pred
    df['l'] = hovertexts

    fig = px.scatter(
        df, x="Average:  (y_true + y_pred) / 2", y="Difference:  y_true - y_pred",
        hover_data='l',
        marginal_x='histogram', marginal_y='histogram',
        title="Bland Altman of True & Predicted Vectors",
        opacity=0.6
    )

    fig.update_traces(marker=dict(color='#ef8354'))
    fig.update_traces(histnorm='probability', selector={'type': 'histogram'}, marker=dict(color='#536b78'))

    if saveToDisk:
        fig.write_html("./blandaltman-true-pred.html")
    else:
        fig.show()
