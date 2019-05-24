import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from Models.LeNet.LeNet import LeNet
from Utils.ExportData import ExportRunningData


class MNISTdata():

    def __init__(self):
        self.data_path = 'data/MNIST'
        self.data_path_processed_train = 'data/MNIST/processed/training.pt'
        self.data_path_processed_test = 'data/MNIST/processed/test.pt'

    def downloadMNISTdata(self):
        datasets.MNIST(self.data_path, train=True, download=True,
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))]))

        datasets.MNIST(self.data_path, train=False, download=True,
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))]))


    def prepareMNISTdata(self,batch_size_train,batch_size_test):


        train_dataset = torchvision.datasets.MNIST(self.data_path,transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = batch_size_train,
                                                        shuffle=True)

        test_dataset = torchvision.datasets.MNIST(self.data_path,train=False,transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

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

def test(network,test_loader,epoch, exportTestData):
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

  exportTestData.addNewData(epoch=epoch,
                             batch=None,
                             running_exanples=epoch*len(test_loader.dataset),
                             loss=test_loss,
                             accuracy=accuracy)
  return correct

def main(download=False):
    n_epochs = 50
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    MNIST = MNISTdata()

    if download:
        MNIST.downloadMNISTdata()

    MNIST.prepareMNISTdata(batch_size_train, batch_size_test)

    exportTestData = ExportRunningData()
    exportTrainData = ExportRunningData()

    network = LeNet(input_layers=1)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)


    resultsPath = 'Results/LeNet MNIST baseline'
    maxCorrect = 0

    for epoch in range(1, n_epochs+1):
        network = train(network,optimizer,MNIST.train_loader,log_interval, epoch,exportTrainData)
        correct = test(network,MNIST.test_loader,epoch,exportTestData)

        if maxCorrect < correct:
            maxCorrect = correct
            print("saving model...")
            torch.save(network.state_dict(), resultsPath + '/model.pth')
            torch.save(optimizer.state_dict(), resultsPath + '/optimizer.pth')


    exportTestData.saveData('Results/LeNet MNIST baseline/TestRunning')
    exportTrainData.saveData('Results/LeNet MNIST baseline/TrainRunning')

def testPreDefinedmodel():
    batch_size_train = 64
    batch_size_test = 1000

    MNIST = MNISTdata()
    continued_network = LeNet(1)

    network_state_dict = torch.load('FinalProject/Results/LeNet MNIST baseline/model.pth')
    continued_network.load_state_dict(network_state_dict)

    MNIST.prepareMNISTdata(batch_size_train, batch_size_test)

    test(continued_network, MNIST.test_loader, [])

if __name__ == '__main__':

    #main()
    testPreDefinedmodel()