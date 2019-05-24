
import torch.nn as nn
import torch.nn.functional as F
import math
from Utils import utils
import Utils.DataTracker

class BasicBranch(nn.Module):
    def __init__(self, input_dims,out_dims):
        super(BasicBranch, self).__init__()

        mid_dims =int(math.ceil(input_dims/out_dims))
        self.fc1 = nn.Linear(input_dims, mid_dims)
        self.fc2 = nn.Linear(mid_dims, out_dims)
        self.fc3 = nn.Linear(out_dims, 1)


    def forward(self, x):
        # out = x.view(x.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)
        #
        # certainty = F.relu(out)
        # certainty = F.sigmoid(self.fc3(certainty))

        out = Utils.DataTracker.DataTracker(x.view(x.size(0), -1))

        out.set_value(self.fc1(out.currentValue))
        out.set_value(F.relu(out.currentValue))
        out.set_value(self.fc2(out.currentValue))

        networkOutput = out.currentValue

        out.set_value(F.relu(out.currentValue))
        out.set_value(F.sigmoid(self.fc3(out.currentValue)))

        forwardOutput = Utils.DataTracker.PathOutput(networkOutput, certainty=out.currentValue,
                                                     outByLayer=out.trackerDict)

        return forwardOutput

class BasicBranch_loss_2(nn.Module):
    def __init__(self, input_dims,out_dims):
        super(BasicBranch_loss_2, self).__init__()

        mid_dims =int(math.ceil(input_dims/out_dims))
        self.fc1 = nn.Linear(input_dims, mid_dims)
        self.fc2 = nn.Linear(mid_dims, out_dims)
        self.fc3 = nn.Linear(mid_dims, 1)


    def forward(self, x):
        # out = x.view(x.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)
        #
        # certainty = F.relu(out)
        # certainty = F.sigmoid(self.fc3(certainty))

        out = Utils.DataTracker.DataTracker(x.view(x.size(0), -1))

        out.set_value(self.fc1(out.currentValue))
        out.set_value(F.relu(out.currentValue))

        valueToCertainty = out.currentValue

        out.set_value(self.fc2(out.currentValue))

        networkOutput = out.currentValue

        #out.set_value(F.relu(out.currentValue))
        out.set_value(F.sigmoid(self.fc3(valueToCertainty)))

        forwardOutput = Utils.DataTracker.PathOutput(networkOutput, certainty=out.currentValue,
                                                     outByLayer=out.trackerDict)

        return forwardOutput

class BasicConvolutionBranch(nn.Module):
    def __init__(self, in_channels,in_channels_dims,out_dims):
        super(BasicConvolutionBranch, self).__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        lastChannelDim = utils.convLayerOutputSize(in_channels_dims, 3,stride=1,padding=1)
        lastChannelDim = utils.convLayerOutputSize(lastChannelDim, 3, stride=2)

        self.unitsInLastfeatures = int(lastChannelDim*lastChannelDim * 32)

        self.classifier = nn.Sequential(
            nn.Linear(self.unitsInLastfeatures, out_dims),
            nn.ReLU(inplace=True),
            nn.Linear(out_dims, 1),
            nn.Sigmoid()
        )


    def forward(self, x):

        out = Utils.DataTracker.DataTracker(x)

        for f in self.features.children():
            out.set_value(f(out.currentValue))

        out.set_value(out.currentValue.view(out.currentValue.size(0), -1))

        for f in self.classifier.children():
            out.set_value(f(out.currentValue))

        #len(out.trackerDict)-4 == nn.Linear(self.unitsInLastfeatures, out_dims) results
        forwardOutput = Utils.DataTracker.PathOutput(out.trackerDict[len(out.trackerDict)-4],
                                                     certainty=out.currentValue,
                                                     outByLayer=out.trackerDict)

        return forwardOutput


class TwoConvolutionBranch(nn.Module):
    def __init__(self, in_channels,in_channels_dims,out_dims):
        super(TwoConvolutionBranch, self).__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        lastChannelDim = utils.convLayerOutputSize(in_channels_dims, 3,stride=1,padding=1) #first conv
        lastChannelDim = utils.convLayerOutputSize(lastChannelDim, 3, stride=1, padding=1) #second conv
        lastChannelDim = utils.convLayerOutputSize(lastChannelDim, 3, stride=2) #pooling

        self.unitsInLastfeatures = int(lastChannelDim*lastChannelDim * 32)

        self.classifier = nn.Sequential(
            nn.Linear(self.unitsInLastfeatures, out_dims),
            nn.ReLU(inplace=True),
            nn.Linear(out_dims, 5), #New addition to certainty training 29.3.19
            nn.ReLU(inplace=True),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )


    def forward(self, x):

        out = Utils.DataTracker.DataTracker(x)

        for f in self.features.children():
            out.set_value(f(out.currentValue))

        out.set_value(out.currentValue.view(out.currentValue.size(0), -1))

        for f in self.classifier.children():
            out.set_value(f(out.currentValue))

        #len(out.trackerDict)-4 == nn.Linear(self.unitsInLastfeatures, out_dims) results
        forwardOutput = Utils.DataTracker.PathOutput(out.trackerDict[len(out.trackerDict)-6],
                                                     certainty=out.currentValue,
                                                     outByLayer=out.trackerDict)

        return forwardOutput

class TwoConvBranchCertaintySeparate(nn.Module):
    def __init__(self, in_channels,in_channels_dims,out_dims):
        super(TwoConvBranchCertaintySeparate, self).__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        lastChannelDim = utils.convLayerOutputSize(in_channels_dims, 3,stride=1,padding=1) #first conv
        lastChannelDim = utils.convLayerOutputSize(lastChannelDim, 3, stride=1, padding=1) #second conv
        lastChannelDim = utils.convLayerOutputSize(lastChannelDim, 3, stride=2) #pooling

        self.unitsInLastfeatures = int(lastChannelDim*lastChannelDim * 32)

        self.classifier = nn.Sequential(
            nn.Linear(self.unitsInLastfeatures, out_dims),
            nn.ReLU(inplace=True),
        )

        certaintyChannelDim = utils.convLayerOutputSize(in_channels_dims, 4, stride=2)  # pooling
        certaintyChannelLinearDim = int(certaintyChannelDim*certaintyChannelDim * in_channels)

        self.certainty = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Linear(certaintyChannelLinearDim, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        out = Utils.DataTracker.DataTracker(x)

        for f in self.features.children():
            out.set_value(f(out.currentValue))

        out.set_value(out.currentValue.view(out.currentValue.size(0), -1))

        for f in self.classifier.children():
            out.set_value(f(out.currentValue))


        forwardOutput = Utils.DataTracker.PathOutput(out.currentValue)


        convLayer = True
        for f in self.certainty.children():

            if convLayer:
                out.set_value(f(x))
                out.set_value(out.currentValue.view(out.currentValue.size(0), -1))
                convLayer =False
            else:
                out.set_value(f(out.currentValue))

        forwardOutput.networkCertainty = out.currentValue

        forwardOutput.outByLayer = out.trackerDict

        return forwardOutput