import torch
import torch.nn as nn

class SOCRNN(nn.Module):
  def __init__(self, num_features, hidden_size, target_size, num_layers = 1):
        super().__init__()
        self.num_features = num_features  # this is the number of features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
        )
        
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=target_size)

  def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        out, hn = self.rnn(x, h)
        out = self.linear(out[:, -1])  
        
        return out