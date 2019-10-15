import torch
import collections
from Utils.ExportData import ExportBranchyNetEval
from Utils import LoadNets
from Models.LeNet.TrainMNIST import MNISTdata
from Models.AlexNet.TrainAlexNet import CIFAR10data
import TrainBranchyModelExpCertWeights
import argparse

def eval(BrancyNetwork, test_loader, exportEvalData, certaintyThreshold=0.5,GroundTruthAvailable = True):
    BrancyNetwork.eval()

    correctByPath = {}
    correctByPathAboveThreshold = {}
    correctByPathBelowThreshold = {}
    totalCases = {}
    totalCasesAboveThreshold = {}
    totalCasesBelowThreshold = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            branchyNetworkOutput = BrancyNetwork(data)

            if GroundTruthAvailable == False:
               target = branchyNetworkOutput['main'].networkOutput.data.max(1, keepdim=True)[1]
               target = torch.squeeze(target)

            casesToCheck = torch.ones(target.data.shape).byte().to(device)

            #seperate main branch output and branches output
            mainOutput = branchyNetworkOutput['main']

            branchyNetworkOutput.pop('main', None)

            #order by branch index
            branchyNetworkOutput = collections.OrderedDict(sorted(branchyNetworkOutput.items()))

            #calculate accuracy for each brach for the following:
            #1. for all data
            #2. for data with Certainty >= certaintyThreshold
            for path, pathOutput in branchyNetworkOutput.items():
                output = pathOutput.networkOutput
                pred = output.data.max(1, keepdim=True)[1]
                branchCertainty = pathOutput.networkCertainty

                #cases that reach the current output and above threshold
                exitInBranch = (branchCertainty >= certaintyThreshold).float() + casesToCheck.view_as(branchCertainty).float()
                exitInBranch = torch.eq(exitInBranch,2)

                # cases that reach the current output and below threshold
                stayInNet = (branchCertainty < certaintyThreshold).float() + casesToCheck.view_as(branchCertainty).float()
                stayInNet = torch.eq(stayInNet, 2)


                #1. calculate overall accuracy
                pred_target_equality = torch.eq(pred, target.data.view_as(pred))
                correctOverall = torch.sum(pred_target_equality).item()
                casesOverall = len(pred_target_equality)

                #2. Calculate accuracy for cases where branchCertainty >= certaintyThreshold
                predExitInBranch = torch.masked_select(pred, exitInBranch.view_as(pred))
                targetExitInBranch = torch.masked_select(target, exitInBranch.view_as(target))

                pred_target_equality = torch.eq(predExitInBranch, targetExitInBranch.data.view_as(predExitInBranch))
                correctExitInBranch = torch.sum(pred_target_equality).item()
                casesExitInBranch = len(pred_target_equality)

                #3. Calculate accuracy for cases where branchCertainty < certaintyThreshold
                predStayInNet = torch.masked_select(pred, stayInNet.view_as(pred))
                targetStayInNet = torch.masked_select(target, stayInNet.view_as(target))

                pred_target_equality = torch.eq(predStayInNet, targetStayInNet.data.view_as(predStayInNet))
                correctStayInNet = torch.sum(pred_target_equality).item()
                casesStayInNet = len(pred_target_equality)

                if path in correctByPath:
                    correctByPath[path] += correctOverall
                    totalCases[path] += casesOverall
                    correctByPathAboveThreshold[path] += correctExitInBranch
                    totalCasesAboveThreshold[path] += casesExitInBranch
                    correctByPathBelowThreshold[path] += correctStayInNet
                    totalCasesBelowThreshold[path] += casesStayInNet
                else:
                    correctByPath[path] = correctOverall
                    totalCases[path] = casesOverall
                    correctByPathAboveThreshold[path] = correctExitInBranch
                    totalCasesAboveThreshold[path] = casesExitInBranch
                    correctByPathBelowThreshold[path] = correctStayInNet
                    totalCasesBelowThreshold[path] = casesStayInNet
                casesToCheck = stayInNet


            path = 'main'
            pred = mainOutput.networkOutput.data.max(1, keepdim=True)[1]
            exitInBranch = casesToCheck

            # 1. calculate overall accuracy
            pred_target_equality = torch.eq(pred, target.data.view_as(pred))
            correctOverall = torch.sum(pred_target_equality).item()
            casesOverall = len(pred_target_equality)

            # 2. Calculate accuracy for cases reaching last exit in net
            predExitInBranch = torch.masked_select(pred, exitInBranch.view_as(pred))
            targetExitInBranch = torch.masked_select(target, exitInBranch.view_as(target))

            pred_target_equality = torch.eq(predExitInBranch, targetExitInBranch.data.view_as(predExitInBranch))
            correctExitInBranch = torch.sum(pred_target_equality).item()
            casesExitInBranch = len(pred_target_equality)

            if path in correctByPath:
                correctByPath[path] += correctOverall
                totalCases[path] += casesOverall
                correctByPathAboveThreshold[path] += correctExitInBranch
                totalCasesAboveThreshold[path] += casesExitInBranch
            else:
                correctByPath[path] = correctOverall
                totalCases[path] = casesOverall
                correctByPathAboveThreshold[path] = correctExitInBranch
                totalCasesAboveThreshold[path] = casesExitInBranch
                correctByPathBelowThreshold[path] = 0
                totalCasesBelowThreshold[path] = 0

    for path in correctByPath:
        if totalCases[path] >0:
            accuracyOverall = 100* correctByPath[path]/totalCases[path]
        else:
            accuracyOverall = 0

        if totalCasesAboveThreshold[path] >0:
            accuracyExitInBranch = 100* correctByPathAboveThreshold[path]/totalCasesAboveThreshold[path]
        else:
            accuracyExitInBranch = 0

        if totalCasesBelowThreshold[path] >0:
            accuracyStayInNet = 100* correctByPathBelowThreshold[path]/totalCasesBelowThreshold[path]
        else:
            accuracyStayInNet = 0

        exportEvalData.addNewData(certaintyThreshold,
                                  path,
                                  BrancyNetwork.neuronsByPath[path],
                                  accuracyOverall,
                                  totalCasesAboveThreshold[path],
                                  accuracyExitInBranch,
                                  totalCasesBelowThreshold[path],
                                  accuracyStayInNet)

def evalBranchyLeNet(resultsPath,lossFunctionID,GroundTruthAvailable = True):
    #set brnachNetwork
    branchyNet = LoadNets.LoadBranchyLeNet(lossFunctionID=lossFunctionID)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    branchyNet.to(device)

    #set test dataset
    MNIST = MNISTdata()
    MNIST.prepareMNISTdata(64, 1000)

    exportEvalData = ExportBranchyNetEval()

    for certainty in [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.85,0.9,0.95,0.99,0.999]:
        eval(branchyNet, MNIST.test_loader, exportEvalData,certainty,GroundTruthAvailable)

    exportEvalData.saveData(resultsPath)

def evalBranchyAlexNet(resultsPath,lossFunctionID,GroundTruthAvailable = True):
    #set brnachNetwork
    branchyNet = LoadNets.LoadBranchyAlexNetCertainty(certaintyOnInput=True,lossFunctionID=lossFunctionID)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    branchyNet.to(device)

    #set test dataset
    CIFAR10 = CIFAR10data()

    CIFAR10.prepareCIFAR10TestDataset(batch_size_test=100)

    exportEvalData = ExportBranchyNetEval()

    certaintyLevels =  [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.85,0.9,0.95,0.99,0.999]
    for certainty in certaintyLevels:
        print("eval alexnet loss function {} certainty {}".format(lossFunctionID,certainty))
        eval(branchyNet, CIFAR10.test_loader, exportEvalData,certainty,GroundTruthAvailable)

    exportEvalData.saveData(resultsPath)

def evalDefalutBranchyAlexNet():
    branchyNet, optimizer, scheduler, CIFAR10 = TrainBranchyModelExpCertWeights.defineNetwork(True, lr=0.01)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    branchyNet.to(device)

    # set test dataset
    CIFAR10 = CIFAR10data()

    CIFAR10.prepareCIFAR10TestDataset(batch_size_test=100)

    exportEvalData = ExportBranchyNetEval()

    certaintyLevels = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                       0.95, 0.99, 0.999]
    for certainty in certaintyLevels:
        print("eval default alexnet certainty {}".format(certainty))
        eval(branchyNet, CIFAR10.test_loader, exportEvalData, certainty, GroundTruthAvailable=True)

    exportEvalData.saveData("Results/eval default network")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--lossFunctionID', type=int, default=2, help='choose loss function for training')
    parser.add_argument('--netArct', default="AlexNet", help='which model to train')
    args = parser.parse_args()

    if args.netArct == "AlexNet":
        if args.lossFunctionID == 1:
            evalBranchyAlexNet('Results/Loss function 1 main to certainty/AlexNetBranch no ground truth',
                               GroundTruthAvailable=False,
                               lossFunctionID=1) #'Results/AlexNetBranch no ground truth/EvalResults'
        if args.lossFunctionID == 2:
            evalBranchyAlexNet('Results/Loss function 2 main to certainty/AlexNetBranch no ground truth',
                               GroundTruthAvailable=False,
                               lossFunctionID=2)
        if args.lossFunctionID == 3:
            evalBranchyAlexNet('Results/Loss function 3 main to certainty/AlexNetBranch no ground truth',
                               GroundTruthAvailable=False,
                               lossFunctionID=3)  # 'Results/AlexNetBranch no ground truth/EvalResults'
        if args.lossFunctionID == 0:
            evalDefalutBranchyAlexNet()
    else:
        if args.lossFunctionID == 1:
            evalBranchyLeNet('Results/Loss function 1 main to certainty/LeNetBranch no ground truth',
                           GroundTruthAvailable=False,
                           lossFunctionID=1)
        if args.lossFunctionID == 2:
            evalBranchyLeNet('Results/Loss function 2 main to certainty/LeNetBranch no ground truth',
                         GroundTruthAvailable=False,
                         lossFunctionID=2)
        if args.lossFunctionID == 3:
            evalBranchyLeNet('Results/Loss function 3 main to certainty/LeNetBranch no ground truth',
                         GroundTruthAvailable=False,
                         lossFunctionID=3)

