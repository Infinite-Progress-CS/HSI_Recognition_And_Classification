import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat
import random
import numpy as np
import torch.nn.functional as F


def to_argment_list(hyperparameter):
    result = [{}]
    arg_ge_one = {}
    for key, arg_list in hyperparameter.items():
        temp_result = []
        if type(arg_list) is not list: arg_list = [arg_list]
        if type(arg_list) is not list: print(8)
        if len(arg_list) > 1:
            arg_ge_one[key] = arg_list
        for di in result:
            for j in arg_list:
                di[key] = j
                temp_result.append(di.copy())
        result = temp_result
    arg_info = "{} argment combination will be run. ".format(len(result)) + "the varying argment:" + str(arg_ge_one)
    print(arg_info)
    return result


def seed_everything(seed=971105):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def loadLabel(path):
    assert os.path.exists(path), '{},路径不存在'.format(path)
    # keys:{train_gt, test_gt}
    gt = loadmat(path)
    return gt['train_gt'], gt['cs_test_gt'], gt['os_test_gt']


class Log(object):
    def __init__(self, path, log_name):
        self.path = path
        self.file_path = os.path.join(path, log_name)
        if os.path.exists(self.file_path):
            os.remove(self.file_path)  # overwrite old log from crashed run
        if not os.path.isdir(path):
            os.makedirs(path)

    
    def record(self, *content, end='\n', sep=''):
        s = ''
        for i, x in enumerate(content):
            if x is not str:
                s += str(content[i])
            else:
                s += x
            if i != len(content) - 1:
                s += sep
        s += end
        with open(self.file_path, 'a+') as f:
            f.write(s)

    def print(self, *content, end='\n', sep=''):
        print(*content, sep=sep)
        s = ''
        for i, x in enumerate(content):
            if x is not str:
                s += str(content[i])
            else:
                s += x
            if i != len(content) - 1:
                s += sep
        s += end
        with open(self.file_path, 'a+') as f:
                    f.write(s)

def ce_loss(logits, targets, reduction='none'):
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:  # 'ce'
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()

def masking(targets, cutoff):
    max_conf, _ = targets.max(dim=1)
    return max_conf > cutoff

def test(model, criterion, dataLoader, use_vit=False, **kwargs):
    device = kwargs.get('device')
    model.eval()
    model.to(device)
    evalLoss, correct = [], 0
    for data in dataLoader:
        input, target = data['x_lb'], data['y_lb']
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            if use_vit:
                logits = model(input)
            else:
                logits = model(input)
        logits = logits.squeeze(-1).squeeze(-1)
        loss = criterion(logits, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)


def infer(model, dataloader, use_vit=False, **kwargs):
    feats, logits, targets = [], [], []
    device = kwargs.get("device")
    model.eval()
    model.to(device)
    for data in dataloader:
        x, y = data['x_lb'].to(device), data['y_lb'].to(device)
        with torch.no_grad():
            if use_vit:
                feat, logit = model(x, with_feat=True)
            else:
                feat, logit = model.backbone(x, with_feat=True)
        feats.append(feat)
        logits.append(logit)
        targets.append(y)
    feats, logits, targets = torch.cat(feats), torch.cat(logits), torch.cat(targets)
    return feats, logits, targets


def infer_allpixel(model, dataloader, **kwargs):
    all_feats, all_logits= [], []
    device = kwargs.get("device")
    model.eval()
    model.to(device)
    for data in dataloader:
        x = data['x_lb'].to(device)
        with torch.no_grad():
            feat, logit = model.backbone(x, with_feat=True)
        all_feats.append(feat)
        all_logits.append(logit)
    all_feats, all_logits = torch.cat(all_feats), torch.cat(all_logits)
    indices = dataloader.dataset.indices
    targets = dataloader.dataset.label.to(device)
    feats = all_feats[indices]
    logits = all_logits[indices]
    return feats, logits, targets, all_feats, all_logits
