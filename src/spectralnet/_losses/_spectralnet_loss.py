import torch
import torch.nn as nn


class SpectralNetLoss(nn.Module):
    def __init__(self, is_normalized: bool = False):
        super(SpectralNetLoss, self).__init__()

        self.is_normalized = is_normalized

    def forward(
        self, W: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        if self.is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D)[:, None]

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        return loss
