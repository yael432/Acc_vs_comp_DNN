import torch
from Models.LeNet.TrainMNIST import MNISTdata
from Models.LeNet.LeNet import LeNet
from Models.BranchyModels.BranchyNet import BranchyNet
from Models.AlexNet.AlexNet import  AlexNetMain
from Models.AlexNet.TrainAlexNet import CIFAR10data


def LoadLeNet():

    pretrainedLeNet = LeNet(1)

    network_state_dict = torch.load('Results/LeNet MNIST baseline/model.pth')
    pretrainedLeNet.load_state_dict(network_state_dict)

    return pretrainedLeNet

def LoadBranchyLeNet(lossFunctionID):
    # set branch layers
    branchPoints = {3: 'basic'}

    MNIST = MNISTdata()
    MNIST.prepareMNISTdata(64, 1000)

    examples = enumerate(MNIST.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    orginalNet = LoadLeNet()

    brancyNet = BranchyNet(orginalNet, branchPoints, example_data)

    if lossFunctionID == 1:
        network_state_dict = torch.load('Results/Loss function 1 main to certainty/LeNetBranch no ground truth/model.pth')
    elif lossFunctionID == 2:
        network_state_dict = torch.load('Results/Loss function 2 main to certainty/LeNetBranch no ground truth/model.pth')
    else:
        network_state_dict = torch.load('Results/Loss function 3 main to certainty/LeNetBranch no ground truth/model.pth')

    brancyNet.load_state_dict(network_state_dict)

    return brancyNet

def loadAlexNet():

    pretrainedLeNet = AlexNetMain(num_classes=10)

    network_state_dict = torch.load('C:\Yael Codes\Pyhton\FinalProject\Results\AlexNet CIFAR baseline\model.pth')
    pretrainedLeNet.load_state_dict(network_state_dict)

    return pretrainedLeNet

def LoadBranchyAlexNet(lossFunctionID):
    batch_size_train = 100
    batch_size_test = 100

    # set branch layers
    branchPoints = {3: 'two_conv'}

    CIFAR10 = CIFAR10data()

    CIFAR10.prepareCIFAR10TestDataset(batch_size_test)

    examples = enumerate(CIFAR10.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    orginalNet = loadAlexNet()

    brancyNet = BranchyNet(orginalNet, branchPoints, example_data)

    if lossFunctionID == 1:
        network_state_dict = torch.load(
            'Results/Loss function 1 exp certainty/AlexNetBranch first no ground truth/model.pth')
    elif lossFunctionID == 2:
        network_state_dict = torch.load(
            'Results/Loss function 2 exp certainty/AlexNetBranch first no ground truth/model.pth')
    else:
        network_state_dict = torch.load(
            'Results/Loss function 3 exp certainty/AlexNetBranch first no ground truth/model.pth')

    brancyNet.load_state_dict(network_state_dict)

    return brancyNet

def LoadBranchyAlexNetCertainty(certaintyOnInput,lossFunctionID):
    batch_size_train = 100
    batch_size_test = 100

    # set branch layers


    if certaintyOnInput:
        branchPoints = {3: 'two_conv_certainty_on_input'}
        if lossFunctionID == 1:
            modelPath =  'Results/Loss function 1 main to certainty/AlexNetBranch no ground truth/model.pth'
        elif lossFunctionID == 2:
            modelPath =  'Results/Loss function 2 main to certainty/AlexNetBranch no ground truth/model.pth'
        else:
            modelPath =  'Results/Loss function 3 main to certainty/AlexNetBranch no ground truth/model.pth'
    else:
        branchPoints = {3: 'two_conv'}
        modelPath = 'Results/Loss function 1 exp certainty/AlexNetBranch no ground truth/model.pth'

    CIFAR10 = CIFAR10data()

    CIFAR10.prepareCIFAR10TestDataset(batch_size_test)

    examples = enumerate(CIFAR10.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    orginalNet = loadAlexNet()

    brancyNet = BranchyNet(orginalNet, branchPoints, example_data)

    network_state_dict = torch.load(modelPath)
    brancyNet.load_state_dict(network_state_dict)

    return brancyNet