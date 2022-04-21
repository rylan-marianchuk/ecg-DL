from sqlite_wrapper import SqliteDBWrap
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

DATASETS_PATH = "/home/rylan/datasets/"
DECODER_PATH = "/home/rylan/PycharmProjects/batch-ecg-receive/database/decoder.db"

def write_xml(filename, desc, modality, metadata, obs_filenames, targets):
    """

    :param filename: (str)
    :param desc: (str)
    :param modality: (str CODE)
    :param metadata: (dict)
    :param obs_filenames: (list str)
    :param targets: (list)
    :return: None, save the file
    """
    return


# Define functions that call self.write_xml. Getting parameters strucutred into this call is depended on how the
# dataset is given externally, and therefore cannot be predicted. It is to be handled ad-hoc / independently

def ejection_fraction(given):
    """

    :param given: (str) path to csv file
    :return: None
    """
    given_df = pd.read_csv(given)
    given_df = given_df[given_df["lvef"].notna()]
    given_df = given_df.drop_duplicates()
    given_df = given_df.sort_values(by=['PUID'])
    puids_given = list(given_df["PUID"])
    lvefs_given = list(given_df["lvef"])
    ecg_dates = list(given_df["ecg_date"])
    sqlwrap = SqliteDBWrap(os.getenv("ECG_DB_PATH") + "decoder.db")

    total = len(puids_given)

    train_size, val_size, test_size = int(total * 0.8), int(total * 0.1), int(total * 0.1)
    euids = set()
    puids_prev = set()
    for indices, name in zip((range(train_size), range(train_size, train_size + val_size), range(train_size + val_size, total)), ("train", "validate", "test")):
        puids = set()
        root = ET.Element("DATASET")
        desc = ET.SubElement(root, "desc").text = "Left-Ventricle-Ejection-Fraction"
        modality = ET.SubElement(root, "modality").text = "ECG"
        continuous_target = ET.SubElement(root, "continuous_target").text = "True"
        n_leads = ET.SubElement(root, "n_leads").text = "8"
        dataset = ET.SubElement(root, "dataset")

        size = 0

        for i in indices:
            date_formatted = ecg_dates[i][5:] + "-" + ecg_dates[i].split("-")[0]
            puid = puids_given[i]
            res = sqlwrap.conx.execute("SELECT EUID FROM Decoder WHERE PUID=:puid AND AcquisitionDate=:acqDate", {"puid":puid, "acqDate": date_formatted})
            res = res.fetchall()
            if len(res) == 0: continue

            for euid in res:
                euid = euid[0]
                if euid in euids or puid in puids_prev: continue
                euids.add(euid)
                puids.add(puid)
                obs = ET.SubElement(dataset, "obs")
                id = ET.SubElement(obs, "id").text = str(size)
                filename_uid = ET.SubElement(obs, "filename_uid").text = euid
                target_v = lvefs_given[i]
                target = ET.SubElement(obs, "target").text = str(target_v)
                size += 1

        puids_prev = puids.copy()
        n_obs = ET.SubElement(root, "n_obs").text = str(size)
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open(DATASETS_PATH + "lvef-" + name + ".xml", "w") as f:
            f.write(xmlstr)



def quality_flag():
    """

    :param given: (str) path to csv file
    :return: None
    """
    artifactDS = SqliteDBWrap("/home/rylan/PycharmProjects/quality/artfactDS.db")
    decoder = SqliteDBWrap(os.getenv("ECG_DB_PATH") + "decoder.db")

    root = ET.Element("DATASET")
    desc = ET.SubElement(root, "desc").text = "QualityFlag"
    modality = ET.SubElement(root, "modality").text = "ECG"
    continuous_target = ET.SubElement(root, "continuous_target").text = "False"
    n_leads = ET.SubElement(root, "n_leads").text = "8"
    dataset = ET.SubElement(root, "dataset")

    size = 0

    bad = artifactDS.conx.execute("SELECT * FROM xmlfilename").fetchall()
    for entry in bad:
        euid = decoder.conx.execute("SELECT EUID FROM decoder WHERE IDENTIFIED_XML=:xml", {"xml":entry[0]}).fetchall()[0][0]
        obs = ET.SubElement(dataset, "obs")
        id = ET.SubElement(obs, "id").text = str(size)
        filename_uid = ET.SubElement(obs, "filename_uid").text = euid
        target = ET.SubElement(obs, "target").text = str(entry[1])
        size += 1

    n_obs = ET.SubElement(root, "n_obs").text = str(size)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(DATASETS_PATH + "artifact-lead-V2.xml", "w") as f:
        f.write(xmlstr)





#quality_flag()
ejection_fraction("./ECG_LVEF.csv")
