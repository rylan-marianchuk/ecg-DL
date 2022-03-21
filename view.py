import plotly.graph_objs as go
from xmlExtract import getLeads
from plotly.subplots import make_subplots
import sqlite3

def viewECG(filename, lead=None, save_to_disk=False):
    path2xmls = "/home/rylan/May_2019_XML/"
    ecg = getLeads(path2xmls + filename, 8)
    lead_names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

    if lead is not None:
        signal = ecg[lead]
        fig = go.Figure(go.Scatter(y=signal, mode='markers', marker=dict(color='red')))

        title = "Electrocardiogram " + "<br>File:  " + filename + "<br>Lead:  " + str(lead)

        fig.update_layout(
            title=title,
            yaxis_title="Amplitude (mV)",
            xaxis_title="Sample Number"
        )

        if save_to_disk:
            fig.write_html("ECG-lead-" + str(lead) + "-" + filename[:-4] + ".html")
        else:
            fig.show()
        return

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=lead_names)

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

    master_title = "Electrocardiogram " + "<br>File:  " + filename

    fig.update_layout(title_text=master_title)

    if save_to_disk:
        fig.write_html("8-LeadECG-" + filename[:-4] + ".html")
    else:
        fig.show()

    return

def viewArtifacted(quality_db_path, tab_batch_size=5):
    conx = sqlite3.connect(quality_db_path)
    res = conx.execute("SELECT EUID FROM quality_flag WHERE QUALITY=1")
    flagged_files = res.fetchall()
    count = 0
    for filename_dot_lead in flagged_files:
        filename, lead = filename_dot_lead[0].split(".")
        viewECG(filename + ".xml", lead=int(lead))
        count += 1
        if count == tab_batch_size:
            input("Continue... ")
            count = 0

