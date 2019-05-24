import TrainBranchyModelExpCertWeights
from Models.AlexNet.TrainAlexNet import CIFAR10data
from data.OnlineData import OnlineExample
import random
import torch
import math
from Utils.ExportData import ExportOnlineTraining
import TrainBranchyModelExpCertWeights

#global process evaluating parameters
trainLastExamplesToCheck = 200
convergenceThreshold = 0.1
onlineTrainingSummary = ExportOnlineTraining
shuffleLevel = 1
certaintyLevels =  [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.85,0.9,0.95,0.99,0.999]
shuffleLevels = [1,5,10,50,500,750,1000,3000,5000,10000,50000]

def prepareData():
    CIFAR10 = CIFAR10data()

    # if download:
    #     CIFAR10.downloadCIFAR10data()

    CIFAR10.prepareCIFAR10TrainDataset(1)
    CIFAR10.prepareCIFAR10TestDataset(1)

    onlineData = []
    for batch_idx, (data, target) in enumerate(CIFAR10.train_loader):
        newExample = OnlineExample(data,target)
        onlineData.append(newExample)

    onlineDataSorted = sorted(onlineData, key=lambda x: x.target)

    if shuffleLevel < 50000:
        # shuffle
        onlineDataSorted_copy = onlineDataSorted[::shuffleLevel]
        random.shuffle(onlineDataSorted_copy)
        onlineDataSorted[::shuffleLevel] = onlineDataSorted_copy



    return onlineDataSorted

def prepareNetwork():
    #load alex branchy net
    brancyNet,optimizer,scheduler,CIFAR10 = TrainBranchyModelExpCertWeights.defineNetwork(True)
    return brancyNet,optimizer

def lossFunction(branchyNetwork,branchyNetworkOutput,target):
    loss, lossByPath, correctByPath, certScorePyPath = \
        TrainBranchyModelExpCertWeights.LossFunctionCrossEntropyAndBCE(branchyNetwork,branchyNetworkOutput,target)
    #3 - path index in current branch cofiguration
    return loss, lossByPath[3], correctByPath[3], certScorePyPath[3]


def trainOnline(branchyNetwork,optimizer,onlineData):

    scorePerBatch = []
    idx = 0
    batchidx = 0
    inferenceTime = 0  # last input index before inference stage
    batchScore = 0
    batchAccuracy = 0
    overallAccuracy = 0

    inferenceAccuracyByCert = {}
    casesToEvalByCert = {}

    for cert in certaintyLevels:
        if cert not in inferenceAccuracyByCert:
            inferenceAccuracyByCert[cert] = 0
            casesToEvalByCert[cert] = 0

    converge = False
    inferencePhase = False

    for example in onlineData:
        idx += 1
        batchidx = math.ceil(idx/trainLastExamplesToCheck)
        optimizer.zero_grad()
        output = branchyNetwork(example.dataTensor)


        target = output['main'].networkOutput.data.max(1, keepdim=True)[1]
        target = torch.squeeze(target)
        loss, lossByPath, correct, certScor = lossFunction(branchyNetwork,
                                                            output,
                                                            target)


        exampleScore = (correct + certScor)/2
        batchScore += exampleScore
        batchAccuracy += correct
        overallAccuracy += correct

        if inferencePhase:
            for cert in certaintyLevels:
                if output.networkCertainty.item() >= cert:
                    inferenceAccuracyByCert[cert] += correct
                    casesToEvalByCert[cert] += 1

        if idx%trainLastExamplesToCheck == 0:

            batchScore = batchScore/trainLastExamplesToCheck
            scorePerBatch.append(batchScore)

            batchAccuracy = batchAccuracy/trainLastExamplesToCheck
            onlineTrainingSummary.saveBatchAcc(shuffle=shuffleLevel,accuracy=batchAccuracy,batch=batchidx)

            batchScore = 0
            batchAccuracy = 0

            if batchidx > 1:
                accuracyDelta = batchScore[batchidx - 1] - batchScore[batchidx - 2]

            else:
                accuracyDelta = 0

            onlineTrainingSummary.saveBatchScoreChg(shuffle=shuffleLevel, batch=batchidx, scoreChg=accuracyDelta)

            if ~inferencePhase & batchidx > 1:

                converge = abs(accuracyDelta) < convergenceThreshold

                if converge:
                    inferencePhase = True
                    inferenceTime = idx
                    onlineTrainingSummary.saveInferenceTime(shuffle=shuffleLevel,inferenceTime=inferenceTime)

        loss.backward()
        optimizer.step()


    for cert in certaintyLevels:
        if cert not in inferenceAccuracyByCert:

            if casesToEvalByCert[cert] > 0:
                inferenceAccuracyByCert[cert] = inferenceAccuracyByCert[cert]/casesToEvalByCert[cert]

            onlineTrainingSummary.saveInferenceAccuracy(shuffle=shuffleLevel,
                                                        certainty=cert,
                                                        expCount=casesToEvalByCert[cert],
                                                        accuracy=inferenceAccuracyByCert[cert])


def trainOnlineMain():


    for sl in shuffleLevels:
        shuffleLevel = sl
        onlineData = prepareData()
        brancyNet, optimizer = prepareNetwork()
        trainOnline(brancyNet,optimizer,onlineData)

    #TBD
    onlineTrainingSummary.saveDataToCSV()
if __name__ == '__main__':
    trainOnlineMain()

