from torch import nn
from einops import rearrange


class DenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list = [128, 64, 32],
        output_size: int = 10,
        batch_norm: bool = True,
    ):
        super().__init__()

        self.model = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.model.append(nn.Linear(input_size, hidden_size))
            else:
                self.model.append(nn.Linear(hidden_sizes[i - 1], hidden_size))
            if batch_norm:
                self.model.append(nn.BatchNorm1d(hidden_size))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = rearrange(x, "b c w h -> b (c w h)")

        return self.model(x)


if __name__ == "__main__":
    import torch

    model = DenseNet()
    input_tensor = torch.randn(32, 1, 28, 28)
    output = model(input_tensor)
    print("input.shape:", input_tensor.shape)  # input.shape: torch.Size([32, 1, 28, 28])
    print("output.shape:", output.shape)  # pythooutput.shape: torch.Size([32, 10])
