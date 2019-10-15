import math
import collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from Models.LeNet.LeNet import LeNet
from Models.BranchyModels.BranchyNet import BranchyNet
from Utils.ExportData import ExportRunningData
from Models.LeNet.TrainMNIST import MNISTdata
from Models.AlexNet.TrainAlexNet import CIFAR10data
from Models.AlexNet.AlexNet import AlexNetMain
import argparse

def DefineBranchyLeNet():
    # prepare data
    MNIST = MNISTdata()

    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    MNIST.prepareMNISTdata(batch_size_train, batch_size_test)

    # load main model
    examples = enumerate(MNIST.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    orginalNet = LeNet(1)

    network_state_dict = torch.load('Results/LeNet MNIST baseline/model.pth')
    orginalNet.load_state_dict(network_state_dict)

    # set branch layers
    branchPoints = {3: 'basic'}
    # branchPoints = {3: 'basic_loss_2'}

    # initialize branchyNet
    random_seed = 1

    torch.manual_seed(random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    brancyNet = BranchyNet(orginalNet, branchPoints, example_data)
    brancyNet.to(device)

    optimizer = optim.SGD(brancyNet.parameters(), lr=learning_rate,
                          momentum=momentum)

    scheduler = None;

    return brancyNet, optimizer, scheduler, MNIST

def defineNetwork(certaintyOnInput,lr=0.01,weight_decay=0,random_seed = 1):
    # set basic hyperparameters

    batch_size_train = 100
    batch_size_test = 100
    learning_rate = lr
    momentum = 0.5


    CIFAR10 = CIFAR10data()

    # if download:
    #     CIFAR10.downloadCIFAR10data()

    CIFAR10.prepareCIFAR10TrainDataset(batch_size_train)
    CIFAR10.prepareCIFAR10TestDataset(batch_size_test)



    orginalNet = AlexNetMain(num_classes=10)

    network_state_dict = torch.load('C:\Yael Codes\Pyhton\FinalProject\Results\AlexNet CIFAR baseline\model.pth')
    orginalNet.load_state_dict(network_state_dict)

    examples = enumerate(CIFAR10.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # set branch layers
    # branchPoints = {3: 'basic_conv'}
    if certaintyOnInput:
        branchPoints = {3: 'two_conv_certainty_on_input'}
    else:
        branchPoints = {3: 'two_conv'}

    # initialize branchyNet

    # torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    brancyNet = BranchyNet(orginalNet, branchPoints, example_data)
    brancyNet.to(device)
    optimizer = optim.SGD(brancyNet.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.75)
    return brancyNet,optimizer,scheduler,CIFAR10

def LossFunctionCrossEntropyAndBCE(branchyNetwork,branchyNetworkOutput,target):

    #branchyNetworkOutput - output result dictionary
    #keys: main or branch name
    #item: for main = output tensor, number of class size,
        # for branch = (output tensor, certainty, value between 0 to 1)

    weightedLoss = 0

    lossByPath = {}
    correctByPath = {}
    certScorePyPath = {}

    for path, pathOutput in branchyNetworkOutput.items():

        if path == 'main':
            continue

        certaintyloss = 0
        # if path=='main':
        #     output = pathOutput
        # else:
        #     output = pathOutput[0]

        pred = pathOutput.networkOutput.data.max(1, keepdim=True)[1]
        pred_target_equality = torch.eq(pred, target.data.view_as(pred))
        correct = torch.sum(pred_target_equality).item()

        if path != 'main':
            branchCertainty = pathOutput.networkCertainty
            certaintylossFunction = torch.nn.BCELoss()
            certaintyloss = certaintylossFunction(branchCertainty, pred_target_equality.float())

        loss = F.cross_entropy(pathOutput.networkOutput, target)
        weight = branchyNetwork.neuronsByPath['main'] - branchyNetwork.neuronsByPath[path]

        if weight > 0:
            weight = math.log10(weight)
        elif path != 'main':
            weight = 0
            print('branch network has more hidden units than main network')
        else:
            weight = 0

        pathLoss = (loss + certaintyloss)*weight

        weightedLoss = weightedLoss + pathLoss

        lossByPath[path] = pathLoss
        correctByPath[path] = correct

        # testScore

        pred_target_inquality = torch.eq(pred_target_equality, 0)

        predEqFlot = pred_target_equality.float()
        predInEqFlot = pred_target_inquality.float()

        a = predEqFlot * branchCertainty
        b = predInEqFlot * (1 - branchCertainty)

        certScorePyPath[path] = torch.sum(a + b).item()

    return weightedLoss,lossByPath,correctByPath,certScorePyPath

def LossFunctionCrossEntropyAndMSE(branchyNetwork,branchyNetworkOutput,target):

    #branchyNetworkOutput - output result dictionary
    #keys: main or branch name
    #item: for main = output tensor, number of class size,
        # for branch = (output tensor, certainty, value between 0 to 1)

    weightedLoss = 0

    lossByPath = {}
    correctByPath = {}
    certScorePyPath = {}

    for path, pathOutput in branchyNetworkOutput.items():

        if path == 'main':
            continue

        certaintyloss = 0
        # if path=='main':
        #     output = pathOutput
        # else:
        #     output = pathOutput[0]

        pred = pathOutput.networkOutput.data.max(1, keepdim=True)[1]
        pred_target_equality = torch.eq(pred, target.data.view_as(pred))
        correct = torch.sum(pred_target_equality).item()

        if path != 'main':
            branchCertainty = pathOutput.networkCertainty
            certaintylossFunction = torch.nn.MSELoss()
            certaintyloss = certaintylossFunction(branchCertainty, pred_target_equality.float())

        loss = F.cross_entropy(pathOutput.networkOutput, target)
        weight = branchyNetwork.neuronsByPath['main'] - branchyNetwork.neuronsByPath[path]

        if weight > 0:
            weight = math.log10(weight)
        elif path != 'main':
            weight = 0
            print('branch network has more hidden units than main network')
        else:
            weight = 0

        pathLoss = (loss + certaintyloss)*weight

        weightedLoss = weightedLoss + pathLoss

        lossByPath[path] = pathLoss
        correctByPath[path] = correct

        # testScore

        pred_target_inquality = torch.eq(pred_target_equality, 0)

        predEqFlot = pred_target_equality.float()
        predInEqFlot = pred_target_inquality.float()

        a = predEqFlot * branchCertainty
        b = predInEqFlot * (1 - branchCertainty)

        certScorePyPath[path] = torch.sum(a + b).item()

    return weightedLoss,lossByPath,correctByPath,certScorePyPath

def LossFunction3(branchyNetwork,
                  branchyNetworkOutput,
                  target,
                  lossHyP):
    # branchyNetworkOutput - output result dictionary
    # keys: main or branch name
    # item: for main = output tensor, certainty always equal 1,
    # for branch = (output tensor, certainty, value between 0 to 1)

    totalLoss = 0
    certScore = 0
    lossByPath = {}
    correctByPath = {}
    certScorePyPath = {}


    for path, pathOutput in branchyNetworkOutput.items():

        if path == 'main':
            continue

        pathOutput = branchyNetworkOutput[path]
        # classification loss without certantity factor
        pred = pathOutput.networkOutput.data.max(1, keepdim=True)[1]
        pred_target_equality = torch.eq(pred, target.data.view_as(pred))
        correct = torch.sum(pred_target_equality).item()

        branchCertainty = pathOutput.networkCertainty

        lossClassification = (F.cross_entropy(pathOutput.networkOutput, target,reduce=False)*branchCertainty).mean()

        lossExitMain = ((1-branchCertainty)*lossHyP).mean()

        weight = branchyNetwork.neuronsByPath['main'] - branchyNetwork.neuronsByPath[path]

        if weight > 0:
            weight = math.log10(weight)
        elif path != 'main':
            weight = 0
            print('branch network has more hidden units than main network')
        else:
            weight = 0

        loss = (lossClassification + lossExitMain)*weight

        lossByPath[path] = loss
        correctByPath[path] = correct

        totalLoss += loss

        #testScore

        pred_target_inquality = torch.eq(pred_target_equality, 0)

        predEqFlot = pred_target_equality.float()
        predInEqFlot = pred_target_inquality.float()

        a = predEqFlot*branchCertainty
        b = predInEqFlot*(1-branchCertainty)

        certScorePyPath[path] = torch.sum(a + b).item()

    return totalLoss, lossByPath, correctByPath,certScorePyPath

def train(branchyNetwork,
            optimizer,
            train_loader,
            log_interval,
            epoch,
            exportTrainData,
            lossFunctionID,
            lossHyP = 0,
            GroundTruthAvailable = True):

    branchyNetwork.train()

    accuracyByPath = {}
    overallCertScoreByPath = {}
    overallLossByPath = {}
    certaintyByPath = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = branchyNetwork(data)

        if GroundTruthAvailable == False:
            target = output['main'].networkOutput.data.max(1, keepdim=True)[1]
            target = torch.squeeze(target)

        if lossFunctionID == 1:
            loss, lossByPath, correctByPath, certScorePyPath = LossFunctionCrossEntropyAndMSE(branchyNetwork, output,
                                                                                              target)
        elif lossFunctionID == 2:
            loss, lossByPath, correctByPath, certScorePyPath = LossFunctionCrossEntropyAndBCE(branchyNetwork, output,
                                                                                              target)
        else:
            loss, lossByPath, correctByPath, certScorePyPath = LossFunction3(branchyNetwork,
                                                                             output,
                                                                             target,
                                                                             lossHyP)

        for path in correctByPath:
            if path in accuracyByPath:
                accuracyByPath[path] += correctByPath[path]
                overallCertScoreByPath[path] += certScorePyPath[path]
                overallLossByPath[path] += lossByPath[path]
                if path != 'main':
                    certaintyByPath[path] += torch.sum(output[path].networkCertainty).item()
            else:
                accuracyByPath[path] = correctByPath[path]
                overallCertScoreByPath[path] = certScorePyPath[path]
                overallLossByPath[path] = lossByPath[path]
                if path != 'main':
                    certaintyByPath[path] = torch.sum(output[path].networkCertainty).item()

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("combined loss {}".format(loss))
            for path in lossByPath:
                print('Train Epoch: {} , Path: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch,path, batch_idx * len(data), len(train_loader.dataset),
                       100 * batch_idx / len(train_loader), lossByPath[path]))
                exportTrainData.addNewData(epoch=epoch,
                                       batch=batch_idx,
                                       running_exanples=(batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)),
                                       loss=lossByPath[path].item(),
                                       accuracy=None,
                                        path=path)

    for path in correctByPath:
        accuracy = 100 * accuracyByPath[path] / len(train_loader.dataset)
        certScore = 100 * overallCertScoreByPath[path] / len(train_loader.dataset)
        if path != 'main':
            certaintyByPath[path] = certaintyByPath[path]/ len(train_loader.dataset)
        else:
            certaintyByPath['main'] = 1
        exportTrainData.addNewData(certScore=certScore,
                                    epoch=epoch,
                                    batch=None,
                                    running_exanples=epoch * len(train_loader.dataset),
                                    loss=overallLossByPath[path].item(),
                                    accuracy=accuracy,
                                    path=path,
                                    avgCertainty=certaintyByPath[path])
    return branchyNetwork

def test(branchyNetwork,
         test_loader,
         epoch,
         exportTestData,
         lossFunctionID,
        lossHyP = 0,
         weight_decay = 0,
         GroundTruthAvailable = True):

    branchyNetwork.eval()
    test_loss = 0

    accuracyByPath = {}
    certaintyByPath = {}
    overallCertScoreByPath = {}
    correctAvg = 0
    certScoreAvg = 0

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = branchyNetwork(data)

            if GroundTruthAvailable == False:
                target = output['main'].networkOutput.data.max(1, keepdim=True)[1]
                target = torch.squeeze(target)

            if lossFunctionID == 1:
                loss, lossByPath, correctByPath,certScorePyPath = LossFunctionCrossEntropyAndMSE(branchyNetwork,output,target)
            elif lossFunctionID == 2:
                loss, lossByPath, correctByPath,certScorePyPath = LossFunctionCrossEntropyAndBCE(branchyNetwork,output,target)
            else:
                loss, lossByPath, correctByPath,certScorePyPath = LossFunction3(branchyNetwork,
                                                                                output,
                                                                                target,
                                                                                lossHyP)

            test_loss += loss.item()

            for path in correctByPath:
                if path in accuracyByPath:
                    accuracyByPath[path] += correctByPath[path]
                    overallCertScoreByPath[path] += certScorePyPath[path]
                    if path != 'main':
                        certaintyByPath[path] += torch.sum(output[path].networkCertainty).item()
                else:
                    accuracyByPath[path] = correctByPath[path]
                    overallCertScoreByPath[path] = certScorePyPath[path]
                    if path != 'main':
                        certaintyByPath[path] = torch.sum(output[path].networkCertainty).item()

    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)

    for path in correctByPath:
        accuracy = 100 * accuracyByPath[path] / len(test_loader.dataset)
        certScore = 100 * overallCertScoreByPath[path] / len(test_loader.dataset)
        if path != 'main':
            certaintyByPath[path] = certaintyByPath[path] / len(test_loader.dataset)
        else:
            certaintyByPath['main'] = 1
        exportTestData.addNewData( certScore=certScore,
                                  epoch=epoch,
                                  batch=None,
                                  running_exanples=epoch * len(test_loader.dataset),
                                  loss=test_loss,
                                  accuracy=accuracy,
                                  path=path,
                                  avgCertainty=certaintyByPath[path],
                                   weight_decay=weight_decay,
                                   lossHyP=lossHyP)
        correctAvg += accuracy
        certScoreAvg += certScore
        print("path {} average certainty is {}".format(path, certaintyByPath[path]))

    correctAvg = correctAvg / len(correctByPath)
    certScoreAvg = certScoreAvg/ len(correctByPath)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy average: {}/{} ({:.2f}%)\n'.format(
        test_loss, correctAvg, len(test_loader.dataset), correctAvg))

    return correctAvg,certScoreAvg

def trainBranchyNet(brancyNet,
                        scheduler,
                        optimizer,
                        dataLoader,
                        resultsPath,
                        exportTrainData,
                        exportTestData,
                        n_epochs = 200,
                        lossFunctionID = 1,
                        lossHyP = 0,
                        weight_decay =0,
                        GroundTruthAvailable = True,
                        download=False):

    log_interval = 10
    maxCorrect = 0



    for epoch in range(1, n_epochs + 1):
        if scheduler:
            scheduler.step()
        brancyNet = train(brancyNet, optimizer, dataLoader.train_loader, log_interval, epoch, exportTrainData,
                          lossFunctionID, lossHyP,GroundTruthAvailable)
        correctAvg,certScoreAvg = test(brancyNet, dataLoader.test_loader, epoch, exportTestData,lossFunctionID,
                                        lossHyP,weight_decay, GroundTruthAvailable)

        overallScore = (correctAvg+certScoreAvg)/2
        if maxCorrect < overallScore:
            maxCorrect = overallScore
            print("saving model...")
            torch.save(brancyNet.state_dict(), resultsPath + '/model.pth')
            torch.save(optimizer.state_dict(), resultsPath + '/optimizer.pth')

        if epoch % 5 == 0:
            print("saving runing data...")
            exportTestData.saveData(resultsPath + '/TestRunning')
            exportTrainData.saveData(resultsPath + '/TrainRunning')

    return maxCorrect;

def BranchTrainingMain(certaintyOnInput, lossFunctionID,netArct = "AlexNet",lossHyP = 0,weight_decay=0):


    exportTestData = ExportRunningData()
    exportTrainData = ExportRunningData()

    if netArct=="AlexNet":
        brancyNet, optimizer,scheduler, dataLoader = defineNetwork(certaintyOnInput,weight_decay)
    else:
        brancyNet, optimizer, scheduler, dataLoader = DefineBranchyLeNet()

    if certaintyOnInput:
        if lossFunctionID==1:
            if netArct == "AlexNet":
                resultsPath = 'Results/Loss function 1 main to certainty/AlexNetBranch no ground truth'
            else:
                resultsPath = 'Results/Loss function 1 main to certainty/LeNetBranch no ground truth'
        elif lossFunctionID==2:
            if netArct == "AlexNet":
                resultsPath = 'Results/Loss function 2 main to certainty/AlexNetBranch no ground truth'
            else:
                resultsPath = 'Results/Loss function 2 main to certainty/LeNetBranch no ground truth'
        else:
            if netArct == "AlexNet":
                resultsPath = 'Results/Loss function 3 main to certainty/AlexNetBranch no ground truth'
            else:
                resultsPath = 'Results/Loss function 3 main to certainty/LeNetBranch no ground truth'
    else:
        resultsPath = 'Results/Loss function 1 exp certainty/AlexNetBranch no ground truth'

    trainBranchyNet(brancyNet=brancyNet,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        dataLoader=dataLoader,
                        exportTrainData=exportTrainData,
                        exportTestData=exportTestData,
                        resultsPath=resultsPath,
                        n_epochs=30,
                        lossFunctionID = lossFunctionID,
                        lossHyP = lossHyP,
                        weight_decay=weight_decay,
                        GroundTruthAvailable=False)

if __name__ == '__main__':
    #BranchTrainingMain(certaintyOnInput=False)
    #BranchTrainingMain(certaintyOnInput=True,lossFunctionID=1)
    #BranchTrainingMain(certaintyOnInput=True, lossFunctionID=2)
    # BranchTrainingMain(certaintyOnInput=True, lossFunctionID=1,netArct="LeNet")
    # BranchTrainingMain(certaintyOnInput=True, lossFunctionID=2,netArct="LeNet")
    #
    # weight_decay_grid = [0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1]
    # for wd in weight_decay_grid:
    #     BranchTrainingMain(certaintyOnInput=True, lossFunctionID=3, netArct="AlexNet",lossHyP = 33.82,weight_decay=wd)

    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--lossFunctionID', type=int, default=2, help='choose loss function for training')
    parser.add_argument('--netArct', default="AlexNet", help='which model to train')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--CompLoss', type=float, default=0, help='Computational loss parameter')
    args = parser.parse_args()

    BranchTrainingMain(certaintyOnInput=True,
                       lossFunctionID = args.lossFunctionID,
                       netArct=args.netArct,
                       lossHyP=args.weight_decay,
                       weight_decay=args.CompLoss)
