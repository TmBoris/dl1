import torch
import torch.nn as nn


class BasicBlockNet(nn.Module):
    '''Arcitecture from HW2 '''
    def __init__(self, n_classes, out_channels):
        super().__init__()

        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2d_res = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=1)
        self.conv2d_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pooling = nn.AvgPool2d(8)

        self.linear = nn.Linear(in_features=800, out_features=n_classes)

    def forward(self, x):
        main_part = self.bn2(self.conv2d_2(self.relu(self.bn1(self.conv2d_1(x)))))
        resid_part = self.conv2d_res(x)
        block_out = self.pooling(self.relu(main_part + resid_part))
        out = self.linear(torch.flatten(block_out, start_dim=1))

        return out
