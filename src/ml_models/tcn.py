import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self,x):
        return x[:, :, :-self.chomp_size]
    
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        y = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.relu2(self.chomp2(self.conv2(y))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(y + res)
    

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.network(x)
        y = y.mean(dim=2)
        return self.output(y)