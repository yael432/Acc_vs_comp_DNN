import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from Models.AlexNet.AlexNet import AlexNetMain
from Utils.ExportData import ExportRunningData


class CIFAR10data():

    def __init__(self):
        self.data_path = 'C:\Yael Codes\Pyhton\FinalProject\data\CIFAR-10'
        #self.data_path_processed_train = 'data/CIFAR-10/processed/training.pt'
        #self.data_path_processed_test = 'data/CIFAR-10/processed/test.pt'
        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToTensor()])

    def downloadCIFAR10data(self):
        datasets.CIFAR10(self.data_path, train=True, download=True,
                                    transform=self.train_transform)

        datasets.CIFAR10(self.data_path, train=False, download=True,
                                    transform=self.test_transform)


    def prepareCIFAR10TrainDataset(self,batch_size_train):


        train_dataset = torchvision.datasets.CIFAR10(self.data_path,transform=self.train_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = batch_size_train,
                                                        shuffle=True)

    def prepareCIFAR10TestDataset(self,batch_size_test):

        test_dataset = torchvision.datasets.CIFAR10(self.data_path,train=False,transform=self.test_transform)

        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size_test,
                                                       shuffle=True)


def train(network,optimizer,train_loader,log_interval, epoch,exportTrainData):
  network.train()

  correct = 0

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)

    pred = output.networkOutput.data.max(1, keepdim=True)[1]
    correct += torch.sum(torch.eq(pred,target.data.view_as(pred))).item()

    loss = F.cross_entropy(output.networkOutput, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100 * batch_idx / len(train_loader), loss.item()))
      exportTrainData.addNewData(epoch=epoch,
                                 batch=batch_idx,
                                 running_exanples=(batch_idx*64) + ((epoch-1)*len(train_loader.dataset)),
                                 loss=loss.item(),
                                 accuracy=None)


  accuracy = 100 * correct / len(train_loader.dataset)
  exportTrainData.addNewData(epoch=epoch,
                             batch=None,
                             running_exanples=epoch*len(train_loader.dataset),
                             loss=loss.item(),
                             accuracy=accuracy)
  return network

def test(network,test_loader,epoch, exportTestData=None):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.cross_entropy(output.networkOutput, target, size_average=False).item()
      pred = output.networkOutput.data.max(1, keepdim=True)[1]
      correct += torch.sum(torch.eq(pred,target.data.view_as(pred))).item()
  test_loss /= len(test_loader.dataset)
  #test_losses.append(test_loss)

  accuracy = 100.00 * correct / len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),accuracy))

  if exportTestData:
    exportTestData.addNewData(epoch=epoch,
                             batch=None,
                             running_exanples=epoch*len(test_loader.dataset),
                             loss=test_loss,
                             accuracy=accuracy)
  return correct

def main(download=False):
    n_epochs = 200
    batch_size_train = 100
    batch_size_test = 100
    learning_rate = 0.001
    #momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    CIFAR10 = CIFAR10data()

    if download:
        CIFAR10.downloadCIFAR10data()

    CIFAR10.prepareCIFAR10TrainDataset(batch_size_train)
    CIFAR10.prepareCIFAR10TestDataset(batch_size_test)

    exportTestData = ExportRunningData()
    exportTrainData = ExportRunningData()

    network = AlexNetMain(num_classes=10)

    #optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          #momentum=momentum)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)

    resultsPath = 'C:\Yael Codes\Pyhton\FinalProject\Results\AlexNet CIFAR baseline'

    maxCorrect = 0

    for epoch in range(1, n_epochs+1):
        scheduler.step(epoch)

        print(optimizer.param_groups[0]['lr'])
        network = train(network,optimizer,CIFAR10.train_loader,log_interval, epoch,exportTrainData)
        correct = test(network,CIFAR10.test_loader,epoch,exportTestData)

        if maxCorrect < correct:
            maxCorrect = correct
            print("saving model...")
            torch.save(network.state_dict(), resultsPath + '\model.pth')
            torch.save(optimizer.state_dict(), resultsPath + '\optimizer.pth')


    exportTestData.saveData(resultsPath + '\TestRunning')
    exportTrainData.saveData(resultsPath + '\TrainRunning')

def testPreDefinedmodel():
    batch_size_train = 100
    batch_size_test = 100

    CIFAR10 = CIFAR10data()
    continued_network = AlexNetMain(num_classes=10)

    network_state_dict = torch.load('C:\Yael Codes\Pyhton\FinalProject\Results\AlexNet CIFAR baseline\model.pth')
    continued_network.load_state_dict(network_state_dict)

    CIFAR10.prepareCIFAR10TrainDataset(batch_size_train)
    CIFAR10.prepareCIFAR10TestDataset(batch_size_test)

    test(continued_network, CIFAR10.test_loader, [])

if __name__ == '__main__':

    #main(download=True)
    main()
    #testPreDefinedmodel()