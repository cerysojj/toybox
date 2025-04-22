import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class MLP1layer(BaseModel):
  def __init__(self, input_size=51529, num_classes=12, layer=2048):
    super(MLP1layer, self).__init__()
    self.d1 = nn.Linear(input_size, layer)
    self.d2 = nn.Linear(layer, num_classes)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = F.relu(self.d1(x))
    x = F.log_softmax(self.d2(x), dim=1)
    return x

class MLP2layer(BaseModel):
  def __init__(self, input_size=51529, num_classes=12, layers=[2048, 1024]):
    super(MLP2layer, self).__init__()
    self.d1 = nn.Linear(input_size, layers[0])
    self.d2 = nn.Linear(layers[0], layers[1])
    self.d3 = nn.Linear(layers[1], num_classes)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = F.relu(self.d1(x))
    x = F.relu(self.d2(x))
    x = F.log_softmax(self.d3(x), dim=1)
    return x

class MLP3layer(nn.Module):
  def __init__(self, input_size=51529, num_classes=12, layers=[2048, 1024, 512]):
    super(MLP3layer, self).__init__()
    self.d1 = nn.Linear(input_size, layers[0])
    self.d2 = nn.Linear(layers[0], layers[1])
    self.d3 = nn.Linear(layers[1], layers[2])
    self.d4 = nn.Linear(layers[2], num_classes)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = F.relu(self.d1(x))
    x = F.relu(self.d2(x))
    x = F.relu(self.d3(x))
    x = F.log_softmax(self.d4(x), dim=1)
    return x
