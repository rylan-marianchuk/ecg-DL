import time

import torch
from transforms import PowerSpec
from decodeLeads import getLeads
from ecgDataset import ecgDataset
import itertools
import math
import torch.nn as nn
from statsmodels.tsa.stattools import acf
from approxEntropy.apentropy_binding import apentropyPY

def areaofPower(ds):
    """

    :param ds:
    :return: dist shape=(N, n_leads) of parameters for each waveform
    """
    dist = torch.zeros(len(ds), ds.n_leads)
    power = PowerSpec(normalize=True)
    for i in range(len(ds)):
        ecg = getLeads(ds.src + "/" + ds.filenames[i], ds.n_leads)
        for j,lead in enumerate((ecg)):
            dist[i,j] = torch.sum(power(lead)).item()
    return dist


def curveLength(ds):
    """
    :return:
    """
    dist = torch.zeros(len(ds), ds.n_leads)
    for i in range(len(ds)-5000):
        ecg = getLeads(ds.src + "/" + ds.filenames[i], ds.n_leads)
        for j,lead in enumerate((ecg)):
            L = 0
            for x in range(lead.shape[0] - 1):
                L += torch.sqrt(1 + (lead[x + 1] - lead[x]) * (lead[x + 1] - lead[x])).item()
            dist[i,j] = L
    return dist


def entropy_of_hist(ds):
    dist = torch.zeros(len(ds), ds.n_leads)
    for i in range(len(ds)):
        ecg = getLeads(ds.src + "/" + ds.filenames[i], ds.n_leads)
        for j,lead in enumerate((ecg)):
            vals, bins = torch.histogram(lead, bins=40, density=True)
            dist[i,j] = -torch.sum(torch.log2(vals) * vals).item()
    return dist


def all_entropy(ds):
    dist = torch.zeros(len(ds), ds.n_leads)
    start = time.time()
    for i in range(len(ds)):
        ecg = getLeads(ds.src + "/" + ds.filenames[i], ds.n_leads).numpy().astype('float64')
        for j,lead in enumerate((ecg)):
            dist[i,j] = apentropyPY(ecg[j], 5000, m=2, r=3)
    print("Total time: " + str(time.time() - start))
    return dist

def weights(signal):
    seg_size = 1250
    segments = 5000 // seg_size
    remainder = 5000 % seg_size
    nlags = 50

    ACFs = torch.zeros(segments, nlags+1)
    cosSim = nn.CosineSimilarity(dim=0)

    for i,x in enumerate(range(0, 5000, seg_size)):
        if i == segments: break
        segment = signal[x:x+seg_size]
        ACF = acf(segment.numpy(), nlags=nlags)
        ACF = torch.from_numpy(ACF)
        ACFs[i] = ACF

    pairwiseM = torch.zeros(segments, segments)

    for i,j in itertools.combinations(range(segments), r=2):
        A1 = ACFs[i]
        A2 = ACFs[j]
        similarity = cosSim(A1, A2)
        if not (0 <= similarity <= 1):
            similarity = min(similarity, 1.00)
        theta = math.acos(similarity)
        pairwiseM[i,j] = theta
        pairwiseM[j,i] = theta

    return torch.sum(pairwiseM, dim=1)

def AreaofAutocorrelationSegs(ds):
    dist = torch.zeros(len(ds), ds.n_leads)
    for i in range(len(ds)):
        ecg = getLeads(ds.src + "/" + ds.filenames[i], ds.n_leads)
        for j,lead in enumerate((ecg)):
            dist[i,j] = weights(lead).sum().item()
    return dist


