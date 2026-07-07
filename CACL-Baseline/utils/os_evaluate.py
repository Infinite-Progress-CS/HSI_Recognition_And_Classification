import torch
import numpy as np
import math

def evaluate(gt, pred, confuse_size):
    confuse_matrix = np.zeros((confuse_size, confuse_size))
    for i in range(confuse_size):
        for j in range(confuse_size):
            num = ((gt == i) & (pred == j)).sum()
            confuse_matrix[i][j] = num
    num_label = np.sum(confuse_matrix)
    acc_per_class = []
    for i in range(confuse_size):
        OA_i = confuse_matrix[i, i] / (np.sum(confuse_matrix, axis=1)[i])
        acc_per_class.append(OA_i)
    AA = np.mean(acc_per_class)
    sum_TP = 0
    for i in range(confuse_size):
        sum_TP += confuse_matrix[i, i]
    accuracy = sum_TP / np.sum(confuse_matrix)
    pa = np.trace(confuse_matrix) / float(num_label)
    pe = np.sum(np.sum(confuse_matrix, axis=0) * np.sum(confuse_matrix, axis=1)) / \
         float(num_label * num_label)
    kappa = (pa - pe) / (1 - pe)
    return acc_per_class, accuracy, AA, kappa

def getClassCenterRepresentation(representation, label, class_num):
    device = representation.device
    dim = representation.shape[1]
    class_center = torch.zeros(class_num, dim, device=device)
    for i in torch.arange(0, class_num):
        indice = (label == i)
        num = indice.sum()
        if num > 0:
            features = representation[indice]
            class_center[i] = features.sum(dim=0) / num
    return class_center


def getClassCenterRepresentation_weight(representation, label, weight, class_num):
    device = representation.device
    dim = representation.shape[1]
    class_center = torch.zeros(class_num, dim, device=device)
    for i in torch.arange(0, class_num):
        indice = (label == i)
        num = indice.sum()
        if num > 0:
            features = representation[indice]
            temp_weight = weight[indice]
            class_center[i] = (features * temp_weight.reshape(temp_weight.shape[0], -1)).sum(dim=0) / temp_weight.sum()
    return class_center


def getDistance(rep, class_center):
    distances = torch.cdist(rep, class_center, p=2) / math.sqrt(class_center.shape[-1])
    return distances


def prototype_evaluate(model, dataLoader, class_center, threshold=3.0, use_vit=False, **kwargs):
    device, known_num = kwargs.get('device'), kwargs.get('known_num')
    model.eval()
    model.to(device)
    class_center = class_center.to(device)
    correct = 0
    gts, preds = [], []
    for data in dataLoader:
        input, target = data['x_lb'], data['y_lb']
        input, target = input.to(device), target.to(device)
        input = input.reshape(input.shape[0], -1, input.shape[-1])
        with torch.no_grad():
            if use_vit:
                feature, logits = model(input, with_feat=True)
            else:
                output = model(input)
                logits = output['logits']  # 闭集分类结果
                feature = output['feat']
        pred = torch.argmax(logits, dim=-1)
        min_distance = getDistance(feature, class_center)
        is_unknown = min_distance > threshold
        non_indices = torch.nonzero(is_unknown)
        pred[non_indices] = known_num
        gts += target.tolist()
        preds += pred.tolist()
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    confuse_evaluate(gts, preds, known_num + 1)
    return acc


def os_evaluate(gt, pred, confuse_size):
    confuse_matrix = np.zeros((confuse_size, confuse_size))
    for i in range(confuse_size):
        for j in range(confuse_size):
            num = ((gt == i) & (pred == j)).sum()
            confuse_matrix[i][j] = num
    num_label = np.sum(confuse_matrix)
    print(confuse_matrix)
    acc_per_class = []
    for i in range(confuse_size):
        OA_i = confuse_matrix[i, i] / (np.sum(confuse_matrix, axis=1)[i])
        acc_per_class.append(OA_i)
        if i < confuse_size - 1:
            print("标签为{}准确率为{}".format(i, OA_i))
        else:
            print("未知类的准确率为{}".format(OA_i))
    AA = np.mean(acc_per_class)
    print("AA：", AA)
    sum_TP = 0
    for i in range(confuse_size):
        sum_TP += confuse_matrix[i, i]
    accuracy = sum_TP / np.sum(confuse_matrix)
    print("OA：{}".format(accuracy))
    pa = np.trace(confuse_matrix) / float(num_label)
    pe = np.sum(np.sum(confuse_matrix, axis=0) * np.sum(confuse_matrix, axis=1)) / \
         float(num_label * num_label)
    kappa = (pa - pe) / (1 - pe)
    print("kappa系数为{}".format(kappa))


def confuse_evaluate(gt, pred, confuse_size):
    confuse_matrix = np.zeros((confuse_size, confuse_size))
    for i in range(confuse_size):
        for j in range(confuse_size):
            num = ((gt == i) & (pred == j)).sum()
            confuse_matrix[i][j] = num
    num_label = np.sum(confuse_matrix)
    acc_per_class = []
    for i in range(confuse_size):
        OA_i = confuse_matrix[i, i] / (np.sum(confuse_matrix, axis=1)[i])
        if i < confuse_size - 1:
            acc_per_class.append(OA_i)
        else:
            acc_per_known_class = acc_per_class.copy()
            acc_per_class.append(OA_i)
            unknown_acc = OA_i
    AA = np.mean(acc_per_class)  # AA
    sum_TP = 0
    for i in range(confuse_size):
        sum_TP += confuse_matrix[i, i]
    OA = sum_TP / np.sum(confuse_matrix)  # OA
    pa = np.trace(confuse_matrix) / float(num_label)
    pe = np.sum(np.sum(confuse_matrix, axis=0) * np.sum(confuse_matrix, axis=1)) / \
         float(num_label * num_label)
    kappa = (pa - pe) / (1 - pe)  # kappa
    
    return acc_per_known_class, unknown_acc, OA, kappa, AA, confuse_matrix
