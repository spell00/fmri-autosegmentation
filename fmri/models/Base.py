
from torch import nn


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.__input_shape = None

    def forward(self):
        exit('The function `forward` needs to be implemented. This is an abstract class.')

    def get_model_name(self):
        exit('`get_model_name` needs to be implemented. This is an abstract class.')

