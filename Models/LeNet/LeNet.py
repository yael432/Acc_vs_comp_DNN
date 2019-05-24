'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import Utils.DataTracker

class LeNet(nn.Module):
    #input_layers = 1 for graysacle
    #input_layers = 3 for rgb
    def __init__(self, input_layers):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(input_layers, 6, 5,padding=2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16*5*5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.features = nn.Sequential(
            nn.Conv2d(input_layers, 6, 5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )
        self.classifier = nn.Sequential(

            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):

        out = Utils.DataTracker.DataTracker(x)

        for f in self.features.children():
            out.set_value(f(out.currentValue))

        out.set_value(out.currentValue.view(out.currentValue.size(0), -1))

        for f in self.classifier.children():
            out.set_value(f(out.currentValue))

        forwardOutput = Utils.DataTracker.PathOutput(out.currentValue,outByLayer=out.trackerDict)

        return forwardOutput