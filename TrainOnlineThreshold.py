import TrainBranchyModelExpCertWeights
from Models.AlexNet.TrainAlexNet import CIFAR10data
from data.OnlineData import OnlineExample
import random
import torch
import math
import numpy as np
from Utils.ExportData import ExportOnlineTraining
import TrainBranchyModelExpCertWeights
from tqdm import tqdm
import matplotlib.pyplot as plt

#global process evaluating parameters
onlieExmaples = 50000
numClasses = 10
trainLastExamplesToCheck = 200
convergenceThreshold = 0.005
onlineTrainingSummary = ExportOnlineTraining(printToLog=False)
certaintyLevels =  [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.85,0.9,0.95,0.99,0.999]
shuffleLevels = [1,50000] #[50000,10000,5000,3000,1000,750,500,50,10,5,1]

def prepareData(shuffleLevel = 50000):
    CIFAR10 = CIFAR10data()

    # if download:
    #     CIFAR10.downloadCIFAR10data()

    CIFAR10.prepareCIFAR10TrainDataset(1)
    CIFAR10.prepareCIFAR10TestDataset(1)

    onlineData = []
    for batch_idx, (data, target) in enumerate(CIFAR10.train_loader):
        newExample = OnlineExample(data,target)
        onlineData.append(newExample)

    if shuffleLevel > 1:
        onlineDataSorted = sorted(onlineData, key=lambda x: x.target)

        if shuffleLevel < 50000:
            # shuffle
            onlineDataSorted_copy = onlineDataSorted[::shuffleLevel]
            random.shuffle(onlineDataSorted_copy)
            onlineDataSorted[::shuffleLevel] = onlineDataSorted_copy
    else:
        onlineDataSorted = onlineData[:]


    return onlineDataSorted

def prepareNetwork():
    #load alex branchy net
    brancyNet,optimizer,scheduler,CIFAR10 = TrainBranchyModelExpCertWeights.defineNetwork(True,lr=0.01)
    return brancyNet,optimizer

def lossFunction(branchyNetwork,branchyNetworkOutput,target):
    loss, lossByPath, correctByPath, certScorePyPath = \
        TrainBranchyModelExpCertWeights.LossFunctionCrossEntropyAndBCE(branchyNetwork,branchyNetworkOutput,target)
    #3 - path index in current branch cofiguration
    #5 - 1 original example + 4 synthetic data
    return loss, lossByPath[3], correctByPath[3]/5, certScorePyPath[3]/5


def trainOnline(branchyNetwork,optimizer,onlineData,shuffleLevel = 50000):

    print('Starting new training: Shuffle level {}'.format(shuffleLevel))

    exploreClassesPerBatch = np.zeros((1,numClasses))
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
        inferenceAccuracyByCert[cert] = 0
        casesToEvalByCert[cert] = 0

    converge = False
    inferencePhase = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for example in tqdm(onlineData):

        idx += 1
        batchidx = math.floor(idx/trainLastExamplesToCheck)

        #optimizer.zero_grad()

        dataAug = torch.zeros([5, example.dataTensor.size()[1], example.dataTensor.size()[2],example.dataTensor.size()[3]], dtype=torch.float)

        dataAug[0,:,:,:] = example.dataTensor
        dataAug[1, :, :, :] = torch.rot90(example.dataTensor, 1, [2, 3])
        dataAug[2, :, :, :] = torch.rot90(example.dataTensor, 2, [2, 3])
        dataAug[3, :, :, :] = torch.rot90(example.dataTensor, 3, [2, 3])
        dataAug[4, :, :, :] = torch.flip(example.dataTensor, [2, 3])
        # plt.imshow(dataAug[4, :, :, :].permute(1, 2, 0))
        # plt.show()

        data= dataAug.to(device)

        output = branchyNetwork(data)



        targetClass = output['main'].networkOutput.data.max(1, keepdim=True)[1]
        targetClass = torch.squeeze(targetClass).unsqueeze(dim=0)

        trueClass = targetClass.data[0,0].item()

        target = torch.empty(5, 1, dtype=torch.long)
        target = target.fill_(trueClass)
        target = torch.squeeze(target)
        target = target.to(device)

        loss, lossByPath, correct, certScor = lossFunction(branchyNetwork,
                                                            output,
                                                            target)

        loss = loss*(1/50000)
        exploreClassesPerBatch[0, trueClass] += 1

        # if inferencePhase:
        #     lossCoefficient = 1- exploreClassesPerBatch[0, trueClass]/np.sum(exploreClassesPerBatch)
        #     loss = loss * lossCoefficient

        exampleScore = (correct + certScor)/2
        batchScore += exampleScore
        batchAccuracy += correct
        overallAccuracy += correct

        if inferencePhase:
            for cert in certaintyLevels:
                if output[3].networkCertainty.data[0,0].item() >= cert:
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
                accuracyDelta = scorePerBatch[batchidx - 1] - scorePerBatch[batchidx - 2]

            else:
                accuracyDelta = 0

            onlineTrainingSummary.saveBatchScoreChg(shuffle=shuffleLevel, batch=batchidx, scoreChg=accuracyDelta)
            onlineTrainingSummary.exportArray(exploreClassesPerBatch, "exploreClassesPerBatch")

            if ~inferencePhase and batchidx > 1:

                converge = abs(accuracyDelta) < convergenceThreshold

                if converge:
                    inferencePhase = True
                    inferenceTime = idx
                    onlineTrainingSummary.saveInferenceTime(shuffle=shuffleLevel,inferenceTime=inferenceTime)

            optimizer.zero_grad()

        loss.backward()
        optimizer.step()


    for cert in certaintyLevels:

        if casesToEvalByCert[cert] > 0:
            inferenceAccuracyByCert[cert] = inferenceAccuracyByCert[cert]/casesToEvalByCert[cert]

        onlineTrainingSummary.saveInferenceAccuracy(shuffle=shuffleLevel,
                                                    certainty=cert,
                                                    expCount=casesToEvalByCert[cert],
                                                    accuracy=inferenceAccuracyByCert[cert])


def trainOnlineMain():


    for shuffleLevel in shuffleLevels:
        onlineData = prepareData(shuffleLevel)
        brancyNet, optimizer = prepareNetwork()
        trainOnline(brancyNet,optimizer,onlineData,shuffleLevel)
        onlineTrainingSummary.saveDataToCSV()

if __name__ == '__main__':
    trainOnlineMain()

