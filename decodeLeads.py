import torch
import base64
import array
import xml.etree.ElementTree as ET

up = torch.nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

def getLeads(path2file, n_leads):
    """
    Decode the waveforms in the xml, compute leads if 12 requested
    Up sample to 500 fs if needed
    :param path2file: (str) path to read xml file
    :return: (tensor) dtype=float32, shape=(8, 5000) the given lead signals of the ECG
    """
    tree = ET.parse(path2file)
    rhythm_wfrm = tree.findall('.//Waveform')[1]
    rhythm_leads = rhythm_wfrm.findall("LeadData")
    fs = int(rhythm_wfrm.find("SampleBase").text)

    ECG_lead_dict = {}

    # Assume xml always has 8 leads
    for lead_ind in range(8):
        lead_xml = rhythm_leads[lead_ind]
        encodedStr = lead_xml.find("WaveFormData").text
        lead_ID = lead_xml.find("LeadID").text
        to_decode = base64.b64decode(encodedStr)
        # TODO Multiply or Divide by 4.88 here?
        T = torch.tensor(array.array('h', to_decode), dtype=torch.float32)
        if fs == 250:
            T = up(T.unsqueeze(0).unsqueeze(0)).flatten()
        ECG_lead_dict[lead_ID] = T

    ECG = torch.zeros(8, 5000)
    for i, key in enumerate(("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")):
        ECG[i] = ECG_lead_dict[key]

    if n_leads == 12:
        # Compute the final leads
        # III
        lead_III = ECG[1] - ECG[0]
        # aVL
        lead_aVL = (ECG[0] - lead_III) / 2
        # aVR
        lead_aVR = -(ECG[0] + ECG[1]) / 2
        # aVF
        lead_aVF = (lead_III + ECG[1]) / 2

        ECG = torch.vstack((ECG, lead_III, lead_aVL, lead_aVR, lead_aVF))


    return ECG


def getNbeats(path2file):
    """
    Extract the generated QRS identifier within the xml,
    :param path2file: (str) path to read xml file
    :return: (int) number of heart beats in the signal
    """
    tree = ET.parse(path2file)
    qrs_times = tree.find(".//QRSTimesTypes").findall(".//QRS")
    return len(qrs_times)
