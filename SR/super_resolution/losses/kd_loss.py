import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class KDLoss(nn.Module):
    def __init__(self, loss_weight=1.0, temperature=0.15):
        super(KDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector.
            S2_fea (List): contain shape (N, L) vector.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach() / self.temperature, dim=1)
            loss_KD_dis += F.kl_div(S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs
