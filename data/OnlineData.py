class OnlineExample():
    def __init__(self, example_data,example_target,batch_idx):
        self.dataTensor = example_data.clone()
        self.LabelTensor = example_target.clone()
        self.target = example_target.item()
        self.exampleKey = batch_idx
