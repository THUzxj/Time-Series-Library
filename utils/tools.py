import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_input_prediction(input, true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    input_len = len(input)
    plt.plot(range(input_len), input, label='Input', linewidth=2)
    plt.plot(range(input_len, input_len + len(true)), true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(range(input_len, input_len + len(preds)), preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def get_anomaly_periods(labels, max_ignore_interval, max_group_length=None):
    changes = []
    for i, label in enumerate(labels[:-1]):
        if label != labels[i+1]:
            changes.append(i+1)

    for i in range(1, len(changes)):
        if (labels[changes[i-1]] == 0 and (changes[i] - changes[i-1] <= max_ignore_interval)):
            labels[changes[i-1]:changes[i]] = 1
    anomaly_periods = []
    for i, label in enumerate(labels):
        if label == 1:
            if len(anomaly_periods) == 0 or anomaly_periods[-1][1] != i or (max_group_length != None and i - anomaly_periods[-1][0] > max_group_length):
                anomaly_periods.append((i, i+1))
            else:
                anomaly_periods[-1] = (anomaly_periods[-1][0], i+1)
    return anomaly_periods

def get_point_labels(anomaly_periods, length):
    pred = np.zeros(length)
    for pred_group in anomaly_periods:
        pred[pred_group[0]:pred_group[1]+1] = 1
    return pred

def anomaly_recall(pred_groups, anomaly_events, time_length):
    pred_labels = get_point_labels(pred_groups, time_length)
    event_hit = []
    for event in anomaly_events:
        if pred_labels[event[0]:event[1]].sum() > 0:
            event_hit.append(1)
        else:
            event_hit.append(0)
    return sum(event_hit) / len(event_hit) if len(event_hit) > 0 else 0
            
def anomaly_precision(pred_groups, anomaly_events, time_length):
    event_labels = get_point_labels(anomaly_events, time_length)
    group_hit = []
    for group in pred_groups:
        if event_labels[group[0]:group[1]].sum() > 0:
            group_hit.append(1)
        else:
            group_hit.append(0)
    return sum(group_hit) / len(group_hit) if len(group_hit) > 0 else 0


def event_f1(gt, pred, check_length=3):
    pred_groups = get_anomaly_periods(pred, check_length, max_group_length=100)
    anomaly_events = get_anomaly_periods(gt, check_length)
    new_pred_groups = []
    for group in pred_groups:
        new_pred_groups.append((max(0, group[0] - check_length), min(len(gt), group[1] + check_length)))
    recall = anomaly_recall(new_pred_groups, anomaly_events, len(gt))
    precision = anomaly_precision(new_pred_groups, anomaly_events, len(gt))
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0, anomaly_events, pred_groups


def visualize_anomaly_detection(gt, pred, check_length, scores = None, threshold = None, name='./pic/test.pdf'):
    plt.figure()
    if scores is not None:
        plt.plot(scores, label='Anomaly Scores', linewidth=2)
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='-')

    pred_groups = get_anomaly_periods(pred, check_length)
    anomaly_events = get_anomaly_periods(gt, check_length)

    ylim = plt.ylim()

    for group in pred_groups:
        plt.fill_between(np.arange(group[0], group[1]), ylim[0], ylim[1], color='r', alpha=0.3, label='Predicted Anomaly')

    for event in anomaly_events:
        plt.fill_between(np.arange(event[0], event[1]), ylim[0], ylim[1], color='g', alpha=0.3, label='Ground Truth Anomaly')
    
    # delete duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(name, bbox_inches='tight')
