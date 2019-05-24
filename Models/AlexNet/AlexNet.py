
import torch.nn as nn
import torch.nn.functional as F
import Utils.DataTracker

class AlexNetMain(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetMain, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        out = Utils.DataTracker.DataTracker(x)

        for f in self.features.children():
            out.set_value(f(out.currentValue))

        out.set_value(out.currentValue.view(out.currentValue.size(0), 1024))

        for f in self.classifier.children():
            out.set_value(f(out.currentValue))

        forwardOutput = Utils.DataTracker.PathOutput(out.currentValue, outByLayer=out.trackerDict)

        return forwardOutput