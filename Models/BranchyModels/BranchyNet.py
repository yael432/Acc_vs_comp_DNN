
import torch.nn as nn
from Models.Branches.Branches import BasicBranch,TwoConvBranchCertaintySeparate, BasicConvolutionBranch, BasicBranch_loss_2
import Utils.DataTracker

#branchpoints = dictionary where key = hidden layer name in main model, item = branch to attached
#currenty only one branch type is available

class BranchyNet(nn.Module):
    def __init__(self,mainNet,branchPoints,input_example):
        super(BranchyNet, self).__init__()

        #dictionary key: forward step index, item: branch object (currently only one branch option)
        self.branchPoints = branchPoints

        self.main = mainNet

        self.outputsByOrderOfExit = list(branchPoints.keys())
        self.outputsByOrderOfExit.sort()
        self.outputsByOrderOfExit.append('main')

        #freeze main network parameters
        for param in self.main.parameters():
            param.requires_grad = False

        #list of branches
        self.branches = nn.ModuleList()

        #get output at each layer according to input example
        #will be used to defined branch first layer input size and output layer size
        mainNetworkOutput = self.main(input_example)
        self.out_dims = mainNetworkOutput.networkOutput.size(1)

        #count the number of nuerons at each path, will be used later for loss function
        self.neuronsByPath = {}

        self.neuronsByPath['main'] = 0

        #The following loop add new branch object for each forward step defined in branchPoints
        #also counts the number of neurns used in each path

        branchIndex = 0
        for id, out in mainNetworkOutput.outByLayer.items():

            out_reshpe = out.view(out.size(0), -1)
            nuberOfneurons = out_reshpe.size(1)

            self.neuronsByPath['main'] += nuberOfneurons

            if id in self.branchPoints:
                input_dims = nuberOfneurons

                if branchPoints[id]== 'basic':
                    newBranch = BasicBranch(input_dims,self.out_dims)
                elif branchPoints[id]== 'basic_loss_2':
                    newBranch = BasicBranch_loss_2(input_dims, self.out_dims)
                elif branchPoints[id]== 'basic_conv':
                    newBranch = BasicConvolutionBranch(in_channels =out.size(1),
                                                       in_channels_dims=out.size(2),
                                                       out_dims=self.out_dims)
                elif branchPoints[id]== 'two_conv':
                    newBranch = BasicConvolutionBranch(in_channels =out.size(1),
                                                       in_channels_dims=out.size(2),
                                                       out_dims=self.out_dims)
                elif branchPoints[id]== 'two_conv_certainty_on_input':
                    newBranch = TwoConvBranchCertaintySeparate(in_channels =out.size(1),
                                                                in_channels_dims=out.size(2),
                                                                out_dims=self.out_dims)
                else:
                    newBranch = BasicBranch(input_dims, self.out_dims)

                self.branches.append(newBranch)

                branchNetworkOutput = newBranch(out) #Utils.DataTracker.PathOutput

                self.neuronsByPath[id] = self.neuronsByPath['main']
                for id_b, out_b in branchNetworkOutput.outByLayer.items():
                    if id_b == 0:
                        continue #skip first datatracker value (main network input to branch)
                    self.neuronsByPath[id] += out_b.view(out_b.size(0), -1).size(1)

                branchIndex +=1

    def forward(self, x):
        results = {}
        branchIndex = 0

        mainOutput = self.main(x)

        for id, out in mainOutput.outByLayer.items():
            if id in self.branchPoints:
                newBranch = self.branches[branchIndex]
                branchOut = newBranch(out)
                results[id] = branchOut
                branchIndex +=1
        results["main"] = mainOutput

        return results

