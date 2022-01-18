import os
import time
import torch
from pydicom import dcmread
import xml.etree.ElementTree as ET
from decodeLeads import getLeads
from artifactNoise import checkZeroVec
import random

class datasetAcquirer:

    def __init__(self, src, dst, target_desc, test_train_split=None, n_leads=8, single_lead_obs=False, filter_artifacts=True, **kwarg_mask):
        """
        :param src: collection of Electrocardiograms in either DICOM or XML filetype #TODO directory or SQL db?
        :param dst: (str) output directory to place datasets (train & test)
        :param target_desc: (str) a high level description of the target for this dataset
        :param test_train_split: (float) if None, do not split the dataset. If float between 0 and 1, apply proportion
                                split and save two datasets.
        :param n_leads: (int) 8 or 12, the number of leads desired. If 12, leads III, aVR, aVF, aVL are computed
        :param single_lead_obs: (bool) whether to count each lead as an independent observation in the dataset
        :param filter_artifacts: (bool) whether to apply automated signal artifact detection, throwing away artifacts
        :param kwarg_mask: (dict) key-value pairs to include in dataset may contain:
            "BUID" : {buid : "", ...}
            "EUID" : {euid : "", ...}
            "PUID" : {puid : "", ...}
            "oldest" : "YYYY-MM-DD"
            "newest" : "YYYY-MM-DD"
            (dictionaries for O(1) lookup)
        """
        self.src = src
        self.dst = dst
        self.test_train_split = test_train_split
        self.filter_artifacts = filter_artifacts
        self.kwarg_mask = kwarg_mask
        self.n_leads = n_leads
        self.single_lead_obs = single_lead_obs

        acquiredDSFile = {}
        acquiredDSFile["n_leads"] = n_leads
        acquiredDSFile["single_lead_obs"] = single_lead_obs
        acquiredDSFile["filter_artifacts"] = filter_artifacts
        acquiredDSFile["kwarg_mask"] = kwarg_mask
        acquiredDSFile["target_desc"] = target_desc
        acquiredDSFile["src"] = src

        train_ds = self.writeDataset()
        encounters = len(train_ds)

        acquiredDSFile["encounters"] = encounters
        acquiredDSFile["n_obs"] = encounters if not single_lead_obs else encounters * n_leads
        acquiredDSFile["dataset"] = train_ds

        if test_train_split is None:
            torch.save(acquiredDSFile, dst + "/" + target_desc)
            return

        test_ds = {}
        acquiredDSFile_test = acquiredDSFile.copy()

        tr_encounters = int(encounters * test_train_split)
        te_encounters = encounters - tr_encounters

        acquiredDSFile["encounters"] = tr_encounters
        acquiredDSFile["n_obs"] = tr_encounters if not single_lead_obs else tr_encounters * n_leads

        for _ in range(te_encounters):
            key = list(train_ds.keys())[random.randint(0, len(train_ds) - 1)]
            val = train_ds.pop(key)
            test_ds[key] = val

        acquiredDSFile_test["encounters"] = te_encounters
        acquiredDSFile_test["n_obs"] = te_encounters if not single_lead_obs else te_encounters * n_leads
        acquiredDSFile_test["dataset"] = test_ds

        torch.save(acquiredDSFile, dst + "/TRAIN-" + "-".join(target_desc.split(" ")))
        torch.save(acquiredDSFile_test, dst + "/TEST-" + "-".join(target_desc.split(" ")))
        return


    def writeDataset(self):
        """
        Iterate through the database and extract the files meeting dataset request criteria
        :return: write the DS (DataSet) as a .pt object
        """
        dataset = {}
        for filename in os.listdir(self.src):
            try:
                if filename[-4:] == ".xml":
                    ecg, tree = self.readxml(filename)
                elif filename[-4:] == ".dcm":
                    ecg, tree = self.readdcm(filename)
                else:
                    continue
            except:
                continue

            try:
                dataset[filename] = self.getTarget(ecg, tree)
            except:
                continue
        return dataset



    def readxml(self, filename):
        tree = ET.parse(self.src + "/" + filename)
        self.filterXml(tree)
        ecg = getLeads(self.src + "/" + filename, self.n_leads)
        if self.filter_artifacts:
            self.filterArtifacts(ecg)
        return ecg, tree


    def readdcm(self, filename):
        with open(self.src + "/" + filename, 'rb') as to_read:
            ds = dcmread(to_read)
        xml_string = ds.EncapsulatedDocument.decode(encoding='utf-8')
        tree = ET.fromstring(xml_string)
        self.filterXml(tree)
        ecg = getLeads(self.src + "/" + filename, self.n_leads)
        if self.filter_artifacts:
            self.filterArtifacts(ecg)
        return ecg, tree


    def filterXml(self, tree):
        """
        Inspect the xml tree and mask the file using keyword arguments, throwing an exception if not acceptable
        This exception will be caught in a try catch
        Possible kwargs for masking defined in constructor
        :param tree: xml tree
        :return: None
        """
        fs = int(tree.findall('.//Waveform')[1].find("SampleBase").text)
        if fs != 250 and fs != 500: raise Exception()

        if "BUID" in self.kwarg_mask:
            buid = tree.find(".//DateofBirth").text
            if buid not in self.kwarg_mask["BUID"]: raise Exception()

        if "EUID" in self.kwarg_mask:
            euid = tree.find(".//PatientFirstName").text
            if euid not in self.kwarg_mask["EUID"]: raise Exception()

        if "PUID" in self.kwarg_mask:
            puid = tree.find(".//PatientID").text
            if puid not in self.kwarg_mask["PUID"]: raise Exception()

        if "oldest" in self.kwarg_mask:
            d = tree.find('.//AcquisitionDate').text
            if d is None: raise Exception()
            # Shift date string to YYYY-MM-DD
            formatted = d.split("-")[-1] + "-" + d[:5]
            if formatted < self.kwarg_mask["oldest"]: raise Exception()

        if "newest" in self.kwarg_mask:
            d = tree.find('.//AcquisitionDate').text
            if d is None: raise Exception()
            # Shift date string to YYYY-MM-DD
            formatted = d.split("-")[-1] + "-" + d[:5]
            if formatted > self.kwarg_mask["newest"]: raise Exception()

        # Reached here, all checks passed.
        return


    def filterArtifacts(self, ecg):
        """
        Throwing an exception if any signal not acceptable
        :param ecg: (tensor) dtype=float32, shape=(8, 5000) the given lead signals of the ECG
        :return: None
        """
        # TODO implement artifact detection algorithm
        if checkZeroVec(ecg): raise Exception()


        return


    def getTarget(self, ecg, tree):
        """
        :param ecg: (tensor) shape=(n_leads, 5000)
        :param tree: xml tree for this ECG
        :return: (tensor) or (list of tensors) of the target. If (list) then single_lead_obs=True, a tensor for each lead
        """
        # TODO Acquire target here
        qrs_times = tree.find(".//QRSTimesTypes").findall(".//QRS")
        if int(qrs_times[-1].find("Time").text) > 5000:
            raise Exception()
        return torch.Tensor([len(qrs_times)])

start = time.time()
acquire = datasetAcquirer(src="/home/rylan/May_2019_XML",
                          dst="/home/rylan/DeepLearningRuns/Beats",
                          test_train_split=0.75,
                          target_desc="Heart Beats in Signal",
                          n_leads=8,
                          single_lead_obs=False,
                          filter_artifacts=True)
print(time.time() - start)
