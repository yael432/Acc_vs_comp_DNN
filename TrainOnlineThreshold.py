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
trainBatchSize = 200
convergenceThreshold = 0.005
onlineTrainingSummary = ExportOnlineTraining(printToLog=False)
certaintyLevels =  [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.85,0.9,0.95,0.99,0.999]
shuffleLevels = [50000,10000,5000,3000,1000,750,500,50,10,5,1]
lossReduction = 1/10000
epochs = 5

def prepareData(shuffleLevel = 50000):
    CIFAR10 = CIFAR10data()

    # if download:
    #     CIFAR10.downloadCIFAR10data()

    CIFAR10.prepareCIFAR10TrainDataset(batch_size_train=1)
    CIFAR10.prepareCIFAR10TestDataset(batch_size_test=1)

    onlineData = []

    for batch_idx, (data, target) in enumerate(CIFAR10.train_loader):
        newExample = OnlineExample(data,target,batch_idx)
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

    onlinedataByKey = {}
    keysSorted = []
    keysRandom = []
    for e in range(len(onlineData)):
        keysSorted.append(onlineDataSorted[e].exampleKey)
        keysRandom.append(onlineData[e].exampleKey)
        onlinedataByKey[onlineData[e].exampleKey] = onlineData[e]

    return onlinedataByKey,keysSorted,keysRandom

def prepareNetwork():
    #load alex branchy net
    brancyNet,optimizer,scheduler,CIFAR10 = TrainBranchyModelExpCertWeights.defineNetwork(True,lr=0.01)
    return brancyNet,optimizer

def lossFunction(branchyNetwork,branchyNetworkOutput,target,batchTrainingSize=1000):
    loss, lossByPath, correctByPath, certScorePyPath = \
        TrainBranchyModelExpCertWeights.LossFunctionCrossEntropyAndBCE(branchyNetwork,branchyNetworkOutput,target)
    #3 - path index in current branch cofiguration
    #5 - 1 original example + 4 synthetic data
    return loss, lossByPath[3], correctByPath[3]/batchTrainingSize, certScorePyPath[3]/batchTrainingSize


def trainOnline(branchyNetwork,optimizer,onlineData,keysSorted,keysRandom,shuffleLevel = 50000):

    print('Starting new training: Shuffle level {}'.format(shuffleLevel))

    exploreClassesPerBatch = np.zeros((1,numClasses))
    scorePerBatch = []
    idx = 0
    batchidx = 0
    idInBatch = 0
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

    #training phase
    batchData = torch.zeros([trainBatchSize*5,3,32,32], dtype=torch.float)
    batchtarget = torch.empty(trainBatchSize*5, 1, dtype=torch.long)
    batchDataIdx = np.arange(trainBatchSize*5)
    np.random.shuffle(batchDataIdx)

    for e in tqdm(range(onlieExmaples)):

        if inferencePhase:
            exampleKey = keysRandom[0]
        else:
            exampleKey = keysSorted[0]

        example = onlineData[exampleKey]

        keysRandom.remove(exampleKey)
        keysSorted.remove(exampleKey)

        idx += 1

        data = example.dataTensor.to(device)

        output = branchyNetwork(data)

        targetClass = output['main'].networkOutput.data.max(1, keepdim=True)[1]
        trueClass = targetClass.data[0, 0].item()

        if inferencePhase:
            pred = output[3].networkOutput.data.max(1, keepdim=True)[1]
            pred = pred.item()

            inferenceCheck = pred == trueClass

            for cert in certaintyLevels:
                if output[3].networkCertainty.data[0,0].item() >= cert:
                    inferenceAccuracyByCert[cert] += inferenceCheck
                    casesToEvalByCert[cert] += 1

        #fill batch data
        currentIdx = batchDataIdx[idInBatch]
        batchData[currentIdx, :, :, :] = example.dataTensor
        batchtarget[currentIdx,0] = trueClass
        idInBatch += 1

        currentIdx = batchDataIdx[idInBatch]
        batchData[currentIdx, :, :, :] = torch.rot90(example.dataTensor, 1, [2, 3])
        batchtarget[currentIdx, 0] = trueClass
        idInBatch += 1

        currentIdx = batchDataIdx[idInBatch]
        batchData[currentIdx, :, :, :] = torch.rot90(example.dataTensor, 2, [2, 3])
        batchtarget[currentIdx, 0] = trueClass
        idInBatch += 1

        currentIdx = batchDataIdx[idInBatch]
        batchData[currentIdx, :, :, :] = torch.rot90(example.dataTensor, 3, [2, 3])
        batchtarget[currentIdx, 0] = trueClass
        idInBatch += 1

        currentIdx = batchDataIdx[idInBatch]
        batchData[currentIdx, :, :, :] = torch.flip(example.dataTensor, [2, 3])
        batchtarget[currentIdx, 0] = trueClass
        idInBatch += 1

        if idx % trainBatchSize == 0:
            #print('Starting training for batch number {}'.format(batchidx))

            batchtarget = torch.squeeze(batchtarget)
            data, target = batchData.to(device), batchtarget.to(device)



            for epoch in range(epochs):
                optimizer.zero_grad()
                output = branchyNetwork(data)
                loss, lossByPath, correct, certScor = lossFunction(branchyNetwork,
                                                                   output,
                                                                   target,
                                                                   batchTrainingSize=trainBatchSize*5)

                loss.backward()
                optimizer.step()

            batchScore = (correct + certScor) / 2
            scorePerBatch.append(batchScore)

            if batchidx > 1:
                accuracyDelta = scorePerBatch[batchidx - 1] - scorePerBatch[batchidx - 2]

            else:
                accuracyDelta = 0

            onlineTrainingSummary.saveBatchScoreChg(shuffle=shuffleLevel, batch=batchidx, scoreChg=accuracyDelta)
            onlineTrainingSummary.saveBatchAcc(shuffle=shuffleLevel, accuracy=correct, batch=batchidx)

            if batchidx > 1:
                converge = abs(accuracyDelta) < convergenceThreshold

                if converge:
                    inferencePhase = True
                    inferenceTime = idx
                    onlineTrainingSummary.saveInferenceTime(shuffle=shuffleLevel,inferenceTime=inferenceTime)

            idInBatch =0
            batchidx += 1
            batchData = torch.zeros([trainBatchSize * 5, 3, 32, 32], dtype=torch.float)
            batchtarget = torch.empty(trainBatchSize * 5, 1, dtype=torch.long)
            batchDataIdx = np.arange(trainBatchSize * 5)
            np.random.shuffle(batchDataIdx)


    for cert in certaintyLevels:

        if casesToEvalByCert[cert] > 0:
            inferenceAccuracyByCert[cert] = inferenceAccuracyByCert[cert]/casesToEvalByCert[cert]

        onlineTrainingSummary.saveInferenceAccuracy(shuffle=shuffleLevel,
                                                    certainty=cert,
                                                    expCount=casesToEvalByCert[cert],
                                                    accuracy=inferenceAccuracyByCert[cert])


def trainOnlineMain():


    for shuffleLevel in shuffleLevels:
        onlineData,keysSorted,keysRandom = prepareData(shuffleLevel)
        brancyNet, optimizer = prepareNetwork()
        trainOnline(brancyNet,optimizer,onlineData,keysSorted,keysRandom,shuffleLevel)
        onlineTrainingSummary.saveDataToCSV()

if __name__ == '__main__':
    trainOnlineMain()

