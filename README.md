Accuracy vs. Computation Cost Tradeoff in distributed Deep Neural Networks
Code ReadMe

1	Dependencies
•	The code is written in Python 3.5.2.
•	The code is based on PyTorch and supports PyTorch 1.0.1
•	Additional requirements: 
•	Torchvision 0.2.1
•	Numpy 1.16.1
•	Pandas 0.23.4
2	Data
Datasets CIFAR10 and MNIST were used. Downloaded using TORCHVISION.DATASETS.
PyTorch datasets documentation: https://pytorch.org/docs/stable/torchvision/datasets.html
3	Train main network
3.1	Train LeNet
Models/LeNet/TrainMNIST.py
3.2	Train AlexNet
Run Models/AlexNet/TrainAlexNet.py
4	Train branch network
4.1	Train offline model
Run TrainBranchyModelExpCertWeights.py – lossFunctionID –netArct --weight_decay –CompLoss
Input parameter for training
Parameter	Optional values
lossFunctionID	1 – Loss function with Regression on certainty value
2 - Loss function with Binary loss on certainty value
3 - Computational and classification loss balance
netArct	“AlexNet” or “LeNet”
weight_decay	Float number to use weigh decay during training, 0 for no weigh decay
CompLoss	Float number, selected empirically define the computational loss when using loss function ID=3 (Computational and classification loss balance)  

4.2	Train online model
4.2.1	Threshold approach
Run TrainOnlineThreshold.py
4.2.2	Ensemble approach
Run TrainOnlineEnsemble.py
5	Eval branch result
Run EvalBranchyModel.py – lossFunctionID –netArct
Input parameter for training
Parameter	Optional values
lossFunctionID	1 – Loss function with Regression on certainty value
2 - Loss function with Binary loss on certainty value
3 - Computational and classification loss balance
0 – Option for AlexNet, evaluate result on default weights on branch (baseline model)
netArct	“AlexNet” or “LeNet”


