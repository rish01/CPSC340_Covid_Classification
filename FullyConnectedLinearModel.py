from torch import nn


class FullyConnectedLinearModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, inp):
        out = self.fc(inp)
        return out


""" 
Option 2

self.fc = nn.Sequential([nn.Linear(resnet_model.fc.in_features, 500),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(500,2)])

"""
