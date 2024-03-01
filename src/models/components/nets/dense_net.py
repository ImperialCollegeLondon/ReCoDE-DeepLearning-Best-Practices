from torch import nn
from einops import rearrange


class DenseNet(nn.Module):
    """
    DenseNet class that extends nn.Module.

    Attributes:
        model (nn.Sequential): Sequential model containing the layers of the network.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list = [128, 64, 32],
        output_size: int = 10,
        batch_norm: bool = True,
    ):
        """
        Initialize DenseNet.

        Args:
            input_size (int): Size of the input. Default is 784.
            hidden_sizes (list): List of sizes for each hidden layer. Default is [128, 64, 32].
            output_size (int): Size of the output. Default is 10.
            batch_norm (bool): Whether to include batch normalization layers. Default is True.
        """
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
        """
        Define the forward pass of the DenseNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, width, height).
            width (int): Width of the input tensor (default is 28).
            height (int): Height of the input tensor (default is 28).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch, output_size).
        """

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
