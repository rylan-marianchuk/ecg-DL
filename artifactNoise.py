import torch
import math
import pywt
from ecgDataset import ecgDataset
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from transforms import Wavelet
from decodeLeads import getNbeats

def checkZeroVec(ecg):
    """
    :param ecg: (tensor) dtype=float32, shape=(8, 5000) the given lead signals of the ECG
    :return: True if this ecg contains a zero vector
    """
    for i in range(ecg.shape[0]):
        if torch.count_nonzero(ecg[i]).item() == 0:
            return True
    return False


def FANE(signal):
    """

    :param signal: tensor of shape=(5000,)
    :return: Frequency-adaptive noise estimator
    """
    fs = 500
    fn = 14
    L = math.floor(math.log2(fs/fn))
    sigma_noise = 0
    detail_coeffs = pywt.wavedec(signal, 'db6', level=L)
    for i in range(L-2, L+1):
        xD = torch.from_numpy(detail_coeffs[-i])
        MAD_estimator = torch.median(torch.abs(xD)).item() / 0.6745
        sigma_noise += MAD_estimator * fs / 100

    return min(sigma_noise, 1)


def middleSVTSize(path2file, signal):
    n_beats = getNbeats(path2file)
    w_u = 175  # is 50 ms
    w_l = 13  # is 25 ms
    step = 1
    variances = []
    for i in range(0, len(signal)-w_u, step):
        window = signal[i:i+w_u]
        v_i = torch.var(window).item()
        variances.append(v_i)

    variances = torch.Tensor(variances)
    # Normalize
    variances -= torch.min(variances)
    variances /= torch.max(variances) - torch.min(variances)

    mu = torch.mean(variances).item()
    sd = torch.std(variances).item()
    T_l = mu - 0.25 * sd
    T_u = mu + 1.5 * sd
    filter_above = variances > T_l
    filter_below = variances < T_u

    both = torch.bitwise_and(filter_above, filter_below)
    return variances[both].sum() / n_beats
    #return torch.count_nonzero(both) / n_beats


def SVT_plot(signal):
    """

    :param signal: tensor of shape=(5000,)
    :return: Frequency-adaptive noise estimator
    """
    w_u = 175  # is 50 ms
    w_l = 13  # is 25 ms
    step = 1
    variances = []
    for i in range(0, len(signal)-w_u, step):
        window = signal[i:i+w_u]
        v_i = torch.var(window).item()
        variances.append(v_i)

    variances = torch.Tensor(variances)
    # Normalize
    variances -= torch.min(variances)
    variances /= torch.max(variances) - torch.min(variances)



    fig = make_subplots(
        rows=2, cols=1)
    fig.add_trace(go.Scatter(y=variances, mode='markers'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=signal, mode='markers', marker=dict(color='red', size=3)),
                  row=2, col=1)
    fig.show()

    trfm = Wavelet(torch.linspace(40, 0.1, 80), output_size=(40, 500), normalize=True)
    trfmed = trfm(variances)
    fig = go.Figure(data=go.Heatmap(z=trfmed[0][0], x=trfm.domain[1], y=trfm.domain[0].flip(0)))
    fig.update_layout(title="Wavelet Transform  -  Wavelet: " + str(trfm.wavelet))
    fig.update_yaxes(title_text="Wavelet scale", type='category')
    fig.update_xaxes(title_text="Time (seconds)", type='category')
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )
    fig.show()

def ApEn(U, m, r) -> float:
    """Approximate_entropy."""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return _phi(m) - _phi(m + 1)

if __name__ == "__main__":
    ds = ecgDataset("/home/rylan/PycharmProjects/newDL/ALL-max-amps.pt")
    #SVT_plot(ds[15146][0][2])
    SVT_plot(ds[14768][0][2])

