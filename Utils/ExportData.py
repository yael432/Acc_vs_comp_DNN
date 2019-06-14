import pandas as pd
import time
import numpy as np
import csv

class ExportRunningData():
    def __init__(self):
        self.exportData = pd.DataFrame(columns=['epoch','batch','running_exanples','loss','accuracy','path','avgCertainty'])

    def addNewData(self,lossHyP=None,certScore=None,epoch=None,
                   batch=None,running_exanples=None,loss=None,
                   accuracy=None,path='main',avgCertainty=None,weight_decay = None):
        self.exportData = self.exportData.append({'lossHyP':lossHyP,
                                                    'certScore':certScore,
                                                    'epoch':epoch,
                                                  'batch':batch,
                                                  'running_exanples':running_exanples,
                                                  'loss':loss,
                                                  'accuracy':accuracy,
                                                  'path':path,
                                                  'avgCertainty':avgCertainty,
                                                  'weight_decay':weight_decay}, ignore_index=True)

    def saveData(self,filePath):

        filePath = filePath + '/ExportRunningData ' + time.strftime("%Y%m%d-%H%M%S")

        self.exportData.to_csv(filePath, sep='\t')

class ExportBranchyNetEval():
    def __init__(self):
        self.exportData = pd.DataFrame(columns=['threshold',
                                                'path',
                                                'pathComputationCost',
                                                'accuracy',
                                                'casesToEval',
                                                'accuracyToEval',
                                                'casesNotToEval',
                                                'accuracyNotToEval'])

    def addNewData(self,threshold=None,path=None,pathComputationCost=None,
                   accuracy=None,casesToEval=None,
                   accuracyToEval=None,casesNotToEval=None,accuracyNotToEval=None):
        self.exportData = self.exportData.append({'threshold':threshold,
                                                  'path':path,
                                                  'pathComputationCost':pathComputationCost,
                                                  'accuracy':accuracy,
                                                  'casesToEval':casesToEval,
                                                  'accuracyToEval':accuracyToEval,
                                                  'casesNotToEval': casesNotToEval,
                                                  'accuracyNotToEval':accuracyNotToEval}, ignore_index=True)

    def saveData(self,filePath):

        filePath = filePath + '/EvalResults/ExportEvalData ' + time.strftime("%Y%m%d-%H%M%S")

        self.exportData.to_csv(filePath, sep='\t')

# class ExportRunningData():
#     def __init__(self):
#                 self.exportData = pd.DataFrame(
#                     columns=['epoch', 'batch', 'running_exanples', 'loss', 'accuracy', 'path', 'avgCertainty'])
#
#     def addNewData(self, lossHyP=None, certScore=None, epoch=None, batch=None, running_exanples=None, loss=None,
#                   accuracy=None, path='main', avgCertainty=None):
#                     self.exportData = self.exportData.append({'lossHyP': lossHyP,
#                                                           'certScore': certScore,
#                                                           'epoch': epoch,
#                                                           'batch': batch,
#                                                           'running_exanples': running_exanples,
#                                                           'loss': loss,
#                                                           'accuracy': accuracy,
#                                                           'path': path,
#                                                           'avgCertainty': avgCertainty}, ignore_index=True)
#
#     def saveData(self, filePath):
#                 filePath = filePath + '/ExportRunningData ' + time.strftime("%Y%m%d-%H%M%S")
#
#                 self.exportData.to_csv(filePath, sep='\t')

class ExportClassificationModelResult():
    def __init__(self):
        self.trainingRes = pd.DataFrame(columns=['Model',
                                      'Fitting time',
                                      'Scoring time',
                                      'Accuracy',
                                      'Precision',
                                      'Recall',
                                      'F1_score',
                                      'AUC_ROC'])

        self.testRes = pd.DataFrame(columns=['Model',
                                          'Accuracy',
                                          'Precision',
                                          'Recall',
                                          'F1_score'])
    def addTrainingRes(self,modelName=None,cv_fit_time=None,cv_score_time=None,cv_accuracy=None,
                       cv_precision=None,cv_recall=None,cv_f1=None,cv_roc=None):

        self.trainingRes =  self.trainingRes.append({'Model': modelName,
                                                'Fitting time': cv_fit_time,
                                                'Scoring time': cv_score_time,
                                                'Accuracy': cv_accuracy,
                                                'Precision': cv_precision,
                                                'Recall': cv_recall,
                                                'F1_score': cv_f1,
                                                'AUC_ROC': cv_roc}, ignore_index=True)

    def addTestRes(self,modelName=None,acc_test=None,prec_test=None,recall_test=None,f1_test=None):
        self.testRes = self.testRes.append({'Model': modelName,
                                            'Accuracy': acc_test,
                                            'Precision': prec_test,
                                            'Recall': recall_test,
                                            'F1_score': f1_test}, ignore_index=True)

    def saveData(self,filePath):

        filePath = filePath + 'training result ' + time.strftime("%Y%m%d-%H%M%S")

        self.trainingRes.to_csv(filePath, sep='\t')

        filePath = filePath + 'test result ' + time.strftime("%Y%m%d-%H%M%S")

        self.testRes.to_csv(filePath, sep='\t')


class ExportOnlineTraining():
    def __init__(self,printToLog = False):
        self.trainAccuracyDF = pd.DataFrame(columns=['Shuffle',
                                                    'Batch',
                                                    'Accuracy'])
        self.scoreChgDF = pd.DataFrame(columns=['Shuffle',
                                               'Batch',
                                                'Score change'])
        self.inferenceAccuracyDF = pd.DataFrame(columns=['Shuffle',
                                                    'Certainty',
                                                    'Output in branch',
                                                    'Accuracy'])


        self.inferenceTimeDF = pd.DataFrame(columns=['Shuffle',
                                                 'inference time'])

        self.printToLog = printToLog
        fileDir = 'Results/online lr/converge/Online result log_' + time.strftime("%Y%m%d-%H%M%S");
        self.logFile = fileDir  # csv.writer(open(fileDir,"a"))



    def saveBatchAcc(self,shuffle,batch,accuracy):

        self.trainAccuracyDF = self.trainAccuracyDF.append({'Shuffle': shuffle,
                                                            'Batch': batch,
                                                            'Accuracy': accuracy}, ignore_index=True)
        if self.printToLog:
            printToLogFile(self.logFile,['saveBatchAcc',shuffle,batch,accuracy])

    def saveBatchScoreChg(self,shuffle,batch,scoreChg):
        self.scoreChgDF = self.scoreChgDF.append({'Shuffle': shuffle,
                                                    'Batch': batch,
                                                    'Score change': scoreChg}, ignore_index=True)

        if self.printToLog:
            printToLogFile(self.logFile, ['saveBatchScoreChg',shuffle,batch,scoreChg])

    def saveInferenceAccuracy(self,shuffle,certainty,expCount,accuracy):
        self.inferenceAccuracyDF = self.inferenceAccuracyDF.append({'Shuffle': shuffle,
                                                                    'Certainty': certainty,
                                                                    'Output in branch': expCount,
                                                                    'Accuracy': accuracy}, ignore_index=True)

        if self.printToLog:
            printToLogFile(self.logFile, ['saveInferenceAccuracy' ,shuffle,certainty,expCount,accuracy])

    def saveInferenceTime(self,shuffle,inferenceTime):
        self.inferenceTimeDF =self.inferenceTimeDF.append({'Shuffle': shuffle,
                                                    'inference time': inferenceTime}, ignore_index=True)

        if self.printToLog:
            printToLogFile(self.logFile, ['saveInferenceTime' ,shuffle,inferenceTime])

    def saveDataToCSV(self):

        fileDir = 'Results/online lr/converge/Online result_' + time.strftime("%Y%m%d-%H%M%S")

        filePath = fileDir + '_ trainAccuracy'
        self.trainAccuracyDF.to_csv(filePath, sep='\t')

        filePath = fileDir + '_ score change'
        self.scoreChgDF.to_csv(filePath, sep='\t')

        filePath = fileDir + '_ inference accuracy'
        self.inferenceAccuracyDF.to_csv(filePath, sep='\t')

        filePath = fileDir + '_ inference time'
        self.inferenceTimeDF.to_csv(filePath, sep='\t')

    def exportArray(self,dataArr,dataDesc):
        filePath = self.logFile + '_' + dataDesc
        f = open(filePath, 'ab')
        np.savetxt(f, dataArr, delimiter=",",fmt='%d')
        f.close()

def printToLogFile(fileName,fields):

    with open(fileName, 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(fields)
