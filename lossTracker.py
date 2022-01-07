import heapq
import torch
import plotly.graph_objs as go
import numpy as np

class LossTracker:
    def __init__(self, dataset, capacity=10, startepoch=0):
        self.capacity = capacity
        self.best = []
        self.worst = []
        self.empty = True
        self.x_best = []
        self.y_best = []
        self.x_worst = []
        self.y_worst = []
        self.ids_best = []
        self.ids_worst = []
        self.trues_best = []
        self.trues_worst = []
        self.preds_best = []
        self.preds_worst = []
        self.ds = dataset
        self.startepoch = startepoch


    def newEpoch(self):
        self.best = []
        self.worst = []
        self.empty = True



    def lossBatch(self, losses, ids, y_true, y_pred):
        sort = torch.argsort(losses)
        if self.empty:
            self.empty = False
            self.best = [(-losses[index].item(), ids[index], y_true[index].item(), y_pred[index].item()) for index in sort[:self.capacity]]
            heapq.heapify(self.best)
            self.worst = [(losses[index].item(), ids[index], y_true[index].item(), y_pred[index].item()) for index in sort[-(self.capacity + 1):]]
            heapq.heapify(self.worst)
            return


        for index in sort[:self.capacity]:
            loss_val = losses[index].item()
            if loss_val < -self.best[0][0]:
                heapq.heappushpop(self.best, (-loss_val, ids[index], y_true[index].item(), y_pred[index].item()))
            else: break

        for index in sort[-(self.capacity + 1):].flip(0):
            loss_val = losses[index].item()
            if loss_val > self.worst[0][0]:
                heapq.heappushpop(self.worst, (loss_val, ids[index], y_true[index].item(), y_pred[index].item()))
            else: break


    def saveEpoch(self, epoch):
        if epoch < self.startepoch: return
        self.x_best += (torch.rand(self.capacity) + epoch).tolist()
        self.x_worst += (torch.rand(self.capacity) + epoch).tolist()
        for i in range(self.capacity):
            self.y_best.append(-self.best[i][0])
            self.ids_best.append(self.best[i][1])
            self.trues_best.append(self.best[i][2])
            self.preds_best.append(self.best[i][3])

            self.y_worst.append(self.worst[i][0])
            self.ids_worst.append(self.worst[i][1])
            self.trues_worst.append(self.worst[i][2])
            self.preds_worst.append(self.worst[i][3])



    def viewLossTrackerPlots(self, highlight_over_n_instances=5, saveToDisk=False):
        self.viewLossTrackerPlot("Worst 20 losses tracked over epochs", self.x_worst, self.y_worst, self.ids_worst, self.trues_worst, self.preds_worst,
                                  highlight_over_n_instances, saveToDisk)

        self.viewLossTrackerPlot("Best 20 losses tracked over epochs", self.x_best, self.y_best, self.ids_best, self.trues_best, self.preds_best,
                                  highlight_over_n_instances, saveToDisk)



    def viewLossTrackerPlot(self, title, x_epoch_noise, y_losses, ids, true, pred, highlight_over_n_instances, saveToDisk):
        """

        :param x_epoch_noise: (list)
        :param y_losses: (list)
        :param ids: (list)
        :param true: (list)
        :param pred: (list)
        :param highlight_over_n_instances: (int)
        :param saveToDisk: (bool)
        :return:
        """
        x_epoch_noise = np.array(x_epoch_noise)
        y_losses = np.log(np.array(y_losses))
        ids = np.array(ids)
        true = np.array(true)
        pred = np.array(pred)

        f = go.Figure()

        colors = ("#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#bb3e03", "#ae2012",
                  "#9b2226", "#9ef01a", "#38b000", "#007200")

        uniques, counts = np.unique(ids, return_counts=True)
        remove = []
        if counts.max() >= highlight_over_n_instances:
            ids_to_highlight = []
            while len(ids_to_highlight) < len(colors) and (True in (counts >= highlight_over_n_instances)):
                argm = np.argmax(counts)
                ids_to_highlight.append(uniques[argm])
                counts[argm] = 0

            for i,id in enumerate((ids_to_highlight)):
                indices = np.argwhere(ids == id)
                remove += indices.flatten().tolist()
                f.add_trace(go.Scatter(x=x_epoch_noise[indices].flatten(),
                                       y=y_losses[indices].flatten(),
                                       mode='lines+markers',
                                       name=id,
                                       marker_color=colors[i],
                                       hovertemplate='%{text}'+
                                                     '<br><b>log10 MSE Loss<b>: %{y:.2f}'+
                                                     '<br>Epoch: %{x:.0f}',
                                       text = ['<i>Pred</i>: {pred:.2f}<br><i>True</i>: {true:.2f}'.format(true=true[i], pred=pred[i]) for i in range(len(indices))]
                                       ))


        remove = np.array(remove)
        if len(remove) > 0:
            pred = np.delete(pred, remove)
            true = np.delete(true, remove)
            ids = np.delete(ids, remove)
            x_epoch_noise = np.delete(x_epoch_noise, remove)
            y_losses = np.delete(y_losses, remove)

        f.add_trace(go.Scatter(x=x_epoch_noise,
                               y=y_losses,
                               mode='markers',
                               name="Others in worst " + str(self.capacity) + " appearing less than " + str(highlight_over_n_instances) + " times",
                               marker_color="#D8D8D8",
                               hovertemplate='%{text}' +
                                             '<br><b>log10 MSE Loss<b>: %{y:.2f}' +
                                             '<br>Epoch: %{x:.0f}',
                               text=['Id: {id}<br><i>Pred</i>: {pred:.2f}<br><i>True</i>: {true:.2f}'.format(true=true[i], pred=pred[i], id=ids[i]) for i in range(len(ids))]

                               ))

        f.update_layout(
            title=title,
            yaxis_title="log10 MSE Loss",
            xaxis_title="Epoch + noise",
            xaxis = dict(
                tickmode='linear',
                tick0=self.startepoch,
                dtick=1
            ),
            hoverlabel_align='right'
        )

        if saveToDisk:
            f.write_html("./" + title[0] + "-loss-tracking.html")
        else:
            f.show()

