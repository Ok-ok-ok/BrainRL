"""
@article{leroy2020communicative,
  title={Communicative Reinforcement Learning Agents for Landmark Detection in Brain Images},
  author={Leroy, Guy and Rueckert, Daniel and Alansary, Amir},
  journal={arXiv preprint arXiv:2008.08055},
  year={2020}
}

Used this paper for inspiration for the network architecture.
"""

import torch
import torch.nn as nn

class CNNPolicy(nn.Module):

    def __init__(self, memory_frames, action_count, init_xavier = True):
        super().__init__()

        self.memory_frames = memory_frames
        self.action_count = action_count
        self.init_xavier = init_xavier

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.prelu_layer1 = nn.PReLU().to(self.device)
        
        self.conv_layer1 = nn.Conv3d(
            in_channels = memory_frames,
            out_channels = 32,
            kernel_size = (5,5,5),
            padding = 1
        ).to(self.device)

        self.pool_layer1 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)

        self.prelu_layer2 = nn.PReLU().to(self.device)

        self.conv_layer2 = nn.Conv3d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (5,5,5),
            padding = 1
        ).to(self.device)

        self.pool_layer2 = nn.MaxPool3d(kernel_size = (2,2,2,)).to(self.device)

        self.prelu_layer3 = nn.PReLU().to(self.device)

        self.conv_layer3 = nn.Conv3d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (4,4,4),
            padding = 1
        ).to(self.device)

        self.pool_layer3 = nn.MaxPool3d(kernel_size = (2,2,2)).to(self.device)

        self.conv_layer4 = nn.Conv3d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3,3),
            padding = 1
        ).to(self.device)

        self.prelu_layer4 = nn.PReLU().to(self.device)

        self.prelu_layer5 = nn.PReLU().to(self.device)
        self.dense_layer1 = nn.Linear(512, 256).to(self.device)
        self.prelu_layer6 = nn.PReLU().to(self.device)
        self.dense_layer2 = nn.Linear(256, 128).to(self.device)
        self.output_layer = nn.Linear(128, self.action_count).to(self.device)
        
        if init_xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, x):
        x = x.to(self.device) / 255.0
        
        x = self.conv_layer1(x)
        x = self.prelu_layer1(x)
        x = self.pool_layer1(x)

        x = self.conv_layer2(x)
        x = self.prelu_layer2(x)
        x = self.pool_layer2(x)

        x = self.conv_layer3(x)
        x = self.prelu_layer3(x)
        x = self.pool_layer3(x)

        x = self.conv_layer4(x)
        x = self.prelu_layer4(x)

        x = x.view(x.size(0), -1)

        x = self.dense_layer1(x)
        x = self.prelu_layer5(x)

        x = self.dense_layer2(x)
        x = self.prelu_layer6(x)

        x = self.output_layer(x)

        return x
    
    def set_training_mode(self, mode: bool):
        if mode:
            self.train()
        else:
            self.eval()