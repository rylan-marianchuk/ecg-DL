import numpy as np
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
        # Vectors to store results
        y_pred = torch.zeros(ds_test.n_obs)
        y_true = torch.zeros(ds_test.n_obs)
        hovertexts = []  # list to allow for ids to be strings
        batch = 0
        # Execute the model on test set, populating the above containers
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

        # Get the rmse loss of the entire test set
        rmeanloss = torch.mean(torch.sqrt((y_true - y_pred) * (y_true - y_pred)))
        print("Test Root Mean loss:  " + str(rmeanloss))
        # Generate prediction plots
        regression_performance(y_true, y_pred, hovertexts, saveToDisk)
        bland_altman(y_true, y_pred, hovertexts, saveToDisk)

    if saveToDisk:
        D = {'ids':hovertexts, 'y_true':y_true, 'y_pred':y_pred}
        torch.save(D, "true-pred-test.pt")
        # Save the dictionary to a csv for viewing, rounding the decimals
        D["y_pred"] = np.around(D["y_pred"], decimals=4)
        frame = pd.DataFrame(D)
        frame.to_csv("true-pred-test.csv", index=False)
    return


def evaluateBinary(te_dataloader, ds_test, model, device, batch_size, saveToDisk=False):
    """
    Execute the test set on a trained model, evaluate its predictive performance for a binary classification problem
    All the same as continuous evaluation, but now using different loss and sigmoid activation
    :param te_dataloader: (DataLoader object from torch.utils.data.dataloader) test set dataloader
    :param ds_test (ecgDataset object) test set dataset
    :param model: trained model, callable given input features
    :param device: cpu or cuda
    :param batch_size: (int)
    :param saveToDisk: (bool) whether to save as html file
    """
    S = torch.nn.Sigmoid()
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
                l = batch * batch_size * ds_test.n_leads
                r = (batch + 1) * batch_size * ds_test.n_leads
            else:
                l = batch * batch_size
                r = (batch + 1) * batch_size
            y_pred[l:r] = S(prediction.flatten())
            y_true[l:r] = y
            hovertexts += ids
            batch += 1

    L = torch.nn.BCEWithLogitsLoss()
    rmeanloss = L(y_pred, y_true)
    print("Test Root Mean loss:  " + str(rmeanloss))

    AUC(y_true, y_pred, saveToDisk=saveToDisk)
    AUPRC(y_true, y_pred, saveToDisk=saveToDisk)

    if saveToDisk:
        D = {'ids': hovertexts, 'y_true': y_true, 'y_pred': y_pred}
        torch.save(D, "true-pred-test.pt")
        # Save the dictionary to a csv for viewing, rounding the decimals
        D["y_pred"] = np.around(D["y_pred"], decimals=4)
        frame = pd.DataFrame(D)
        frame.to_csv("true-pred-test.csv", index=False)
    return


def AUPRC(y_true, y_score, saveToDisk=False):
    """
    Generate a Precision Recall Curve
    :param y_true: binary
    :param y_score: sigmoid probabilities
    :param saveToDisk: (bool) whether to save as html file
    """
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
        labels=dict(x='False Positive Rate', y='True Positive Rate')
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
    return


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
    return


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
    return


def compare_binary(true_pred_file, threshold, dataset):
    """

    :param true_pred_file: "true-pred-test/train.pt" to acquire the two groups of signals
    :param threshold: (float) in (0, 1) the decision boundary to divide probability outputs into two classes
    """
    id_true_pred_dict = torch.load(true_pred_file)

    ds = torch.load(dataset)
    ds_filenames = np.array(list(ds["dataset"].keys()))

    ids = np.array(id_true_pred_dict["ids"], dtype=np.int_)
    pred = np.array(id_true_pred_dict["y_pred"])

    class_0_ids = ids[pred <= threshold]
    class_0_filenames = ds_filenames[class_0_ids]
    class_1_ids = ids[pred > threshold]
    class_1_filenames = ds_filenames[class_1_ids]


compare_binary("/home/rylan/DeepLearningRuns/Upsample/MayoAge/true-pred-test.pt", 0.5, "/home/rylan/DeepLearningRuns/Upsample/TEST-Whether-the-signal-was-upsampled.pt")
