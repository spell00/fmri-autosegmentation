# Class from https://github.com/ex4sperans/variational-inference-with-normalizing-flows/blob/master/flow.pyimport numpy as np
from torch import nn
import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.classification import (
    stat_scores_multiple_classes
)

def safe_log(z):
    import torch
    return torch.log(z + 1e-7)


class FreeEnergyBound(nn.Module):

    def __init__(self):
        super().__init__()
        self.density = p_z

    def forward(self, zk, log_jacobians):
        sum_of_log_jacobians = sum(log_jacobians)
        KLD = torch.mean(sum_of_log_jacobians - safe_log(self.density(zk)))
        return KLD

def QuantileLoss(preds, target, quantiles):
    def _tilted_loss(q, e):
        return torch.max((q-1) * e, q * e).unsqueeze(1)

    err = target - preds
    losses = [_tilted_loss(q, err[:, i])  # calculate per quantile
              for i, q in enumerate(quantiles)]

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss




class MCC(Metric):
    r"""
    Computes `Mathews Correlation Coefficient <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_:

    Forward accepts
    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``
    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.
    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.
    Args:
        labels: Classes in the dataset.
        pos_label: Treats it as a binary classification problem with given label as positive.
    """

    def __init__(
            self,
            labels,
            pos_label=None,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.labels = labels
        self.num_classes = len(labels)
        self.idx = None

        if pos_label is not None:
            self.idx = labels.index(pos_label)

        self.add_state("matthews_corr_coef", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        tps, fps, tns, fns, _ = stat_scores_multiple_classes(
            pred=preds, target=target, num_classes=self.num_classes)

        if self.idx is not None:
            tps, fps, tns, fns = tps[self.idx], fps[self.idx], tns[self.idx], fns[self.idx]

        numerator = (tps * tns) - (fps * fns)
        denominator = torch.sqrt(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)))

        self.matthews_corr_coef = numerator / denominator
        # Replacing any NaN values with 0
        self.matthews_corr_coef[torch.isnan(self.matthews_corr_coef)] = 0

        self.total += 1

    def compute(self):
        """
        Computes Matthews Correlation Coefficient over state.
        """
        return self.matthews_corr_coef / self.total
