import torch.nn as nn
import torch

class MayoAge(nn.Module):

    def __init__(self):
        """
        Model implemented at publication DOI: 10.1161/CIRCEP.119.007284
        """
        super(MayoAge, self).__init__()
        self.model = nn.Sequential(
            # Temporal 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 7)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Temporal 2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),

            # Temporal 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Temporal 4
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),

            # Temporal 5
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Temporal 6
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Temporal 7
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Temporal 8
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Spatial 1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        :param x: batch of float32 images with one channel, shape=(N, 1, n_leads, 5000)
        :return: scalar prediction
        """
        return self.model(x)


class Wu(nn.Module):

    def __init__(self):
        """
        Model implemented at publication DOI: 10.1109/ACCESS.2019.2956050
        """
        super(Wu, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,5), stride=(2,2)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(2, 2)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(2, 2)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(2, 2)),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(10752, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        :param x: batch of float32 images with one channel, shape=(N, 1, H, W)
                  Note must change size of first linear layer according to H W
        :return: scalar prediction
        """
        return self.model(x)


class FullyConnected(nn.Module):

    def __init__(self):
        super(FullyConnected, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.Linear(2500, 750),
            nn.ReLU(),
            nn.Linear(750, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        :param x: batch of float32 tensors, shape=(N, 5000)
        :return: scalar prediction
        """
        return self.model(x)


class InceptionBlock(nn.Module):
    """
    Inception block module used for the AF model
    """

    def __init__(self, n_in_channels):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(n_in_channels, 32, kernel_size=(1, 40), padding=(0, 15)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(n_in_channels, 32, kernel_size=(1, 20), padding=(0, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(n_in_channels, 32, kernel_size=(1, 10)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 1, 1))
        return

    def forward(self, x):
        """
        :param x: volume tensor of shape=(N, n_in_channels, H, W)
        :return: max pooled volume
        """
        appended = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x)), dim=1)
        return self.maxpool(appended)


class AF(nn.Module):

    def __init__(self):
        """
        Model implemented at publication DOI: 10.1161/CIRCULATIONAHA.120.047829
        See Supplementary document for visual figure of architecture
        """
        super(AF, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(1, 32, (1, 80)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(32, 64, (1, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.inception_block = InceptionBlock(32)

        self.globalavgpool = nn.AvgPool3d(kernel_size=(10, 1, 25))

        self.FC = nn.Sequential(
            nn.Linear(4980, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, b1, b2, b3):
        """
        :param b1: shape=(1, 4, 2496) first 5 seconds of leads I, II, V1, V5
        :param b2: shape=(1, 5, 1248) seconds 5-7.5 of leads V1, V2, V3, II, V5
        :param b3: shape=(1, 5, 1248) seconds 7.5-10 of leads V4, V5, V6, II, V1
        :return: scalar prediction
        """
        b1 = self.first_conv(b1)
        for _ in range(4):
            b1 = self.inception_block(b1)
        b1 = self.second_conv(b1)

        b2 = self.first_conv(b2)
        for _ in range(4):
            b2 = self.inception_block(b2)
        b2 = self.second_conv(b2)

        b3 = self.first_conv(b3)
        for _ in range(4):
            b3 = self.inception_block(b3)
        b3 = self.second_conv(b3)

        # GAP & flatten & concatenate
        gap_b1 = self.globalavgpool(b1).flatten(start_dim=1)
        gap_b2 = self.globalavgpool(b2).flatten(start_dim=1)
        gap_b3 = self.globalavgpool(b3).flatten(start_dim=1)
        return self.FC(torch.cat((gap_b1, gap_b2, gap_b3), dim=1))


def print_sizes(model, input_tensor):
    print("Total parameters to train " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    output = input_tensor
    for m in model.children():
        output = m(output)
        print(m, output.shape)
    return output

# See the shapes at each step of input
if __name__ == "__main__":
    model = Wu().model
    #print(model(torch.rand(15, 1, 4, 2496), torch.rand(15, 1, 5, 1248), torch.rand(15, 1, 5, 1248)))
    #input_tensor = torch.rand(15, 1, 4, 2500), torch.rand(15, 1, 5, 1250), torch.rand(15, 1, 5, 1250)
    #print(print_sizes(model, input_tensor))


