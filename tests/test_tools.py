import pytest
import numpy as np
from utils.tools import event_f1

def test_event_f1():
    gt = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    precision, recall, f1_score, anomaly_events, pred_groups = event_f1(gt, pred, 3)
    assert precision == 1.0
    assert recall == 1.0
    assert f1_score == 1.0
    assert anomaly_events == [(3, 12)]
    assert pred_groups == [(3, 12)]
