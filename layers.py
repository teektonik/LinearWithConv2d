import torch
import torch.nn as nn


class LinearWithConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        FC слой через Conv2d.

        Параметры:
        - in_features (int): количество входных признаков.
        - out_features (int): количество выходных признаков.
        """
        super(LinearWithConv2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Параметры:
        - x (torch.Tensor): входной тензор.

        Выход:
        - torch.Tensor: выходной тензор.
        """
        x = x.unsqueeze(-1).unsqueeze(-1)
        out = self.conv(x)
        return out.squeeze(-1).squeeze(-1)

  
class Linear_(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        FC слой.

        Параметры:
        - in_features (int): количество входных признаков.
        - out_features (int): количество выходных признаков.
        """ 
        super(Linear_, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Параметры:
        - x (torch.Tensor): входной тензор.

        Выход:
        - torch.Tensor: выходной тензор.
        """
        out = self.linear(x)
        return out