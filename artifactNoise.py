import torch
import math
import pywt
from ecgDataset import ecgDataset
import plotly.graph_objs as go
import numpy as np

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


def SVT_plot(signal):
    """

    :param signal: tensor of shape=(5000,)
    :return: Frequency-adaptive noise estimator
    """
    w_u = 25  # is 50 ms
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
    T_l = mu - 0.15 * sd
    T_u = mu + 1.5 * sd
    fig = go.Figure(go.Scatter(y=variances, mode='markers'))
    fig.add_hline(y=T_u)
    fig.add_hline(y=T_l)
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
    ds = ecgDataset("ALL-max-amps.pt")
    SVT_plot(ds[15146][0][2])

