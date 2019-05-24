class DataTracker():


    def __init__(self, initialValue = None):
        self.currentValue = initialValue
        self.trackerDict = {}
        self.countChanges = 0

        self.trackerDict[self.countChanges] = self.currentValue


    def set_value(self, value):
        self.currentValue = value
        self.countChanges +=1
        self.trackerDict[self.countChanges] = self.currentValue

    def getDataTracker(self):
        return self.trackerDict

class PathOutput():
    def __init__(self,output,certainty=1,outByLayer=None):
        self.networkOutput = output
        self.networkCertainty = certainty
        self.outByLayer = outByLayer