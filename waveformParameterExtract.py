import torch
from transforms import PowerSpec
from decodeLeads import getLeads
import xml.etree.ElementTree as ET
from ecgDataset import ecgDataset

def areaofPower(ds):
    """

    :param ds:
    :return: dist shape=(N, n_leads) of parameters for each waveform
    """
    dist = torch.zeros(len(ds), ds.n_leads)
    power = PowerSpec(normalize=True)
    for i in range(len(ds)):
        ecg = getLeads(ET.parse(ds.src + "/" + ds.filenames[i]), ds.n_leads)
        for j,lead in enumerate((ecg)):
            dist[i,j] = torch.sum(power(lead)).item()
    return dist

def curveLength(ds):
    """
    :return:
    """
    dist = torch.zeros(len(ds), ds.n_leads)
    for i in range(len(ds)-5000):
        ecg = getLeads(ET.parse(ds.src + "/" + ds.filenames[i]), ds.n_leads)
        for j,lead in enumerate((ecg)):
            L = 0
            for x in range(lead.shape[0] - 1):
                L += torch.sqrt(1 + (lead[x + 1] - lead[x]) * (lead[x + 1] - lead[x])).item()
            dist[i,j] = L
    return dist

def entropy_of_hist(ds):
    dist = torch.zeros(len(ds), ds.n_leads)
    for i in range(len(ds)):
        ecg = getLeads(ET.parse(ds.src + "/" + ds.filenames[i]), ds.n_leads)
        for j,lead in enumerate((ecg)):
            vals, bins = torch.histogram(lead, bins=40, density=True)
            dist[i,j] = -torch.sum(torch.log2(vals) * vals).item()
    return dist

ds = ecgDataset("ALL-max-amps.pt")
#entropy_of_hist(ds[235][0][0])
#ds.viewByParameter(areaofPower(ds), "Area of Power Spectrum")
print()
