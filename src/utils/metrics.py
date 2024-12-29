# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ------------------------------------------------------------------------------
class Metric(ABC):
    running_sum: float
    count: int

    def __init__(self):
        self.running_sum = 0.0
        self.count = 0

    @abstractmethod
    def update(self, logits, labels, *args, **kwargs):
        pass

    def average(self):
        return self.running_sum / self.count

# ------------------------------------------------------------------------------
class AccuracyMetric(Metric):

    def __init__(self):
        super(AccuracyMetric, self).__init__()

    def update(self, logits, labels, *args, **kwargs):
        _, pred = torch.max(logits.data, 1)
        self.running_sum += pred.eq(labels.view_as(pred)).sum().item()
        self.count += labels.size(0)

# ------------------------------------------------------------------------------
class LossMetric(Metric):

    def __init__(self, criterion=None):
        super(LossMetric, self).__init__()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion

    def update(self, logits, labels, *args, **kwargs):
        self.running_sum += self.criterion(
            logits, labels, *args, **kwargs
        ).item()
        self.count += labels.size(0)

# ------------------------------------------------------------------------------
METRIC_FUNCTION_DICT = {
    'accuracy'  : AccuracyMetric(),
}

# ------------------------------------------------------------------------------
def t1_acc(logits, labels, mask=None):

    _batch, _len = logits.shape
    if mask is None:
        mask = torch.ones(
            labels.shape, dtype=labels.dtype, device=labels.device
        )

    _pred_token = torch.argmax(logits, dim=-1)
    _hit = (_pred_token == labels) * mask

    _t1_acc = torch.zeros_like(labels)
    _all_acc = torch.zeros_like(labels)
    
    for _b in range(0, _batch):
        for _i in range(0, _len):
            if mask[_b, _i] >= 1.0:
                if _hit[_b, _i] > 0:
                    _t1_acc[_b] = 1.0
                break  

        _is_succ = True
        for _i in range(0, _len):
            if mask[_b, _i] >= 1.0:
                if _hit[_b, _i] <= 0:
                    _is_succ = False
                    break

        if _is_succ:
            _all_acc[_b] = 1.0

    #_t1_acc = _t1_acc * 1.0 / _batch
    #_all_acc = _all_acc * 1.0 / _batch
    return _t1_acc, _all_acc

# ------------------------------------------------------------------------------