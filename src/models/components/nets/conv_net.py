import torch.nn as nn
from einops import rearrange


class ConvNet(nn.Module):
    def __init__(self, in_channels=1, channel_sizes=[16, 32, 64], output_size=10, hidden_size=64):
        super(ConvNet, self).__init__()

        self.channel_sizes = channel_sizes
        self.num_layers = len(channel_sizes)

        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i in range(self.num_layers):
            out_channels = channel_sizes[i]

            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.relu_layers.append(nn.ReLU())
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(out_channels, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = self.relu_layers[i](x)
            x = self.pool_layers[i](x)

        x = self.global_avg_pool(x)

        x = rearrange(x, "b c w h -> b (c w h)")

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    import torch

    model = ConvNet()
    input_tensor = torch.randn(32, 1, 28, 28)
    output = model(input_tensor)
    print("input.shape:", input_tensor.shape)  # input.shape: torch.Size([32, 1, 28, 28])
    print("output.shape:", output.shape)  # pythooutput.shape: torch.Size([32, 10])
