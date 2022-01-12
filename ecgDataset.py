import torch
from decodeLeads import getLeads
from torch.utils.data import Dataset
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

class ecgDataset(Dataset):

    def __init__(self, acquiredDSFile, transform=None):
        """
        Dataset is ordered
        :param acquiredDSFile: (str) to the .pt datatset
        :param transform: the by-lead transform to apply in __getitem__()
        """
        self.acquiredDSFile = torch.load(acquiredDSFile)
        self.src = self.acquiredDSFile["src"]
        self.n_leads = self.acquiredDSFile["n_leads"]
        self.single_lead_obs = self.acquiredDSFile["single_lead_obs"]
        self.filenames = list(self.acquiredDSFile["dataset"].keys())
        self.targets = list(self.acquiredDSFile["dataset"].values())
        self.transform = transform
        self.encounters = self.acquiredDSFile["encounters"]
        self.n_obs = self.acquiredDSFile["n_obs"]
        self.lead_names = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6") if self.n_leads == 8 \
            else ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "III", "aVL", "aVR", "aVF")
        self.prediction_assigned = False
        self.predictions = torch.Tensor([])
        self.prediction_ids = []

    def __len__(self):
        return self.acquiredDSFile["encounters"]


    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.singleidx(idx)
        else:
            return self.numerousidx(idx)


    def f2i(self, filestring, returnAsString=True):
        """
        f2i: filename to index
        :param filestring: xml file name, i.e. MUSE_20190315_120232_20000.xml
        :return: index of its place in the dataset, as string or int.
        """
        if returnAsString:
            return str(self.filenames.index(filestring))
        return self.filenames.index(filestring)


    def singleidx(self, id):
        ecg = getLeads(self.src + "/" + self.filenames[id], self.n_leads)
        if self.transform is not None:
            ecg = self.transform(ecg)
        if self.single_lead_obs:
            s_id = str(id)
            return ecg, torch.stack(self.targets[id]).flatten(), [s_id + " " + leadID for leadID in self.lead_names]
        return ecg, self.targets[id], id


    def numerousidx(self, idx):
        for id in idx:
            return self.singleidx(id)


    def denoteArtifact(self, index, leadid):
        """
        Write to file containing all found artifacts
        """
        df = pd.DataFrame()
        df["filename"] = [self.filenames[index]]
        df["lead"] = [leadid]
        with open('artifacts.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)


    def assignPredictions(self, preds):
        """
        :param preds: (tensor) of model outputs, shape=(n_obs,)
        """
        pred_true_file = torch.load(preds)
        self.predictions = pred_true_file["y_pred"]
        self.prediction_ids = pred_true_file["ids"]
        self.prediction_assigned = True


    def viewECG(self, index, leadid=None, saveToDisk=False):
        """

        :param index:
        :param leadid:
        :return:
        """
        ecg = getLeads(self.src + "/" + self.filenames[index], self.n_leads)
        targets = self.targets[index]
        filename = self.filenames[index]

        if leadid is not None:
            lead_index = self.lead_names.index(leadid)
            fig = go.Figure(go.Scatter(y=ecg[lead_index], mode='markers', marker=dict(color='red')))
            title = "Electrocardiogram of dataset index <b>" + str(index) + " " + leadid + \
                    "</b><br>File:  " + filename + \
                    "<br>Target:  " + str(targets[lead_index])
            if self.prediction_assigned:
                v = self.prediction_ids.index(str(index) + " " + leadid)
                title += "<br>Prediction:  " + str(self.predictions[v])

            fig.update_layout(
                title=title,
                yaxis_title="Amplitude (mV)",
                xaxis_title="Sample Number"
            )

            if saveToDisk:
                fig.write_html("ECG-" + filename[:-4] + "-index-" + str(index) + "-" + leadid + ".html")
            else:
                fig.show()
            return

        subplot_titles = [self.lead_names[i] for i in range(self.n_leads)]
        if self.single_lead_obs:
            for i in range(self.n_leads):
                subplot_titles[i] += "<br>Target:  " + str(targets[i])
            if self.prediction_assigned:
                for i in range(self.n_leads):
                    v = self.prediction_ids.index(str(index) + " " + self.lead_names[i])
                    subplot_titles[i] += "<br>Prediction:  " + str(self.predictions[v])

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=subplot_titles)

        fig.add_trace(go.Scatter(y=ecg[0], mode='markers', marker=dict(color='red')),
            row=1, col=1)

        fig.add_trace(go.Scatter(y=ecg[1], mode='markers', marker=dict(color='red')),
            row=1, col=2)

        fig.add_trace(go.Scatter(y=ecg[2], mode='markers', marker=dict(color='red')),
            row=2, col=1)

        fig.add_trace(go.Scatter(y=ecg[3], mode='markers', marker=dict(color='red')),
            row=2, col=2)

        fig.add_trace(go.Scatter(y=ecg[4], mode='markers', marker=dict(color='red')),
            row=3, col=1)

        fig.add_trace(go.Scatter(y=ecg[5], mode='markers', marker=dict(color='red')),
            row=3, col=2)

        fig.add_trace(go.Scatter(y=ecg[6], mode='markers', marker=dict(color='red')),
            row=4, col=1)

        fig.add_trace(go.Scatter(y=ecg[7], mode='markers', marker=dict(color='red')),
            row=4, col=2)


        master_title = "Electrocardiogram of dataset index <b>" + str(index) + \
                       "</b><br>File:  " + filename

        if not self.single_lead_obs:
            master_title += "<br>Target:  " + str(targets[index])
            if self.prediction_assigned:
                master_title += "<br>Prediction:  " + str(self.predictions[index])

        fig.update_layout(title_text=master_title)

        if saveToDisk:
            fig.write_html("8-LeadECG-" + filename[:-4] + "-index-" + str(index) + ".html")
        else:
            fig.show()


    def viewByParameter(self, dist, parameterName, saveToDisk=False):
        """
        :param dist: the parameters extracted from the waveforms, visualize sorted on an axis
                     each row is in same order as self.filenames
                     shape=(N, n_leads) if single_lead_obs
                     shape=(N, 1) else
        :param parameterName: (str) description of the parameter computed for each ECG
        :return: None, visualize using plotly
        """
        hovertexts = []
        x = torch.zeros(dist.shape[0] * dist.shape[1])
        for i in range(dist.shape[0]):
            for j,lead in enumerate((self.lead_names)):
                hovertexts.append(str(i) + " " + lead)
                x[i*self.n_leads + j] = dist[i,j]

        f = go.Figure(data=[go.Scatter(x=x, y=torch.rand(dist.shape[0] * dist.shape[1]),
                                       hovertext=hovertexts, mode='markers', marker=dict())])
        f.update_layout(
            title="Distribution of " + parameterName + " Values",
            yaxis_title="Noise",
            xaxis_title=parameterName
        )
        if saveToDisk:
            f.write_html("viewByParameter-" + parameterName + ".html")
        else:
            f.show()


    def viewTargetDistribution(self, saveToDisk=False):
        """
        Only fit for Regression
        # TODO allow classification
        :return:
        """
        hovertexts = []
        x = torch.zeros(len(self) * self.n_leads)

        for i in range(len(self)):
            for j,lead in enumerate((self.lead_names)):
                hovertexts.append(str(i) + " " + lead)
                x[i*self.n_leads + j] = self.targets[i][j].item()

        f = go.Figure(data=[go.Scatter(x=x, y=torch.rand(len(self) * self.n_leads),
                                       hovertext=hovertexts, mode='markers', marker=dict())])
        f.update_layout(
            title="Distribution of Target " + self.acquiredDSFile["target_desc"] + " Values",
            yaxis_title="Noise",
            xaxis_title=self.acquiredDSFile["target_desc"]
        )
        if saveToDisk:
            f.write_html("ScatterTargetDistribution-" + self.acquiredDSFile["target_desc"] + ".html")
        else:
            f.show()

        fig = go.Figure(data=go.Violin(y=x, box_visible=True, line_color='black',
                                       meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                                       x0='Target Distribution:  ' + self.acquiredDSFile["target_desc"]))
        fig.update_layout(
            title="Distribution of Target " + self.acquiredDSFile["target_desc"] + " Violin Plot",
        )

        if saveToDisk:
            fig.write_html("ViolinTargetDistribution-" + self.acquiredDSFile["target_desc"]  + ".html")
        else:
            fig.show()

