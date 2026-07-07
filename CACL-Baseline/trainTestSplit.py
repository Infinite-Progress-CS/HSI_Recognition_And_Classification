import numpy as np
from scipy.io import loadmat
import os
import random
from scipy import io
from Dataset.HSIDataset import DatasetInfo
import json
TRAIN_SIZE = [10, 20]
RUN = 10 
dir = None

def load(dname):
    info = DatasetInfo.info[dname]
    path = os.path.join("data", dname, info['gt_file_name'])
    dataset = loadmat(path) 
    key = info['label_key'] 
    gt = dataset[key]  # 取出HSI_gt
    return gt, info['class_num']


def save_sample(train_gt, cs_test_gt, os_test_gt, dataset_name, sample_size, run):
    sample_dir = dir + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt': train_gt,
                             'cs_test_gt': cs_test_gt,
                             'os_test_gt': os_test_gt})


def sample_train_test(datasetName, gt, train_size, known_num, class_num):
    train_gt = np.zeros_like(gt)
    os_test_gt = np.copy(gt)
    train_indices = []
    for c in range(1, class_num + 1):
        if datasetName == 'Indian':
            if c > known_num:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            
            print("class {} total sample num {} sampling num {}".format(c, len(X), train_size))
            
            if c == 7 or c == 9:
                # 如果为第7或第9类，则固定取10个样本
                train_indices += random.sample(X, 10)  
            else:
                # 当样本数目大于20时且类别为1, 4, 13, 16时，只选择10个样本。
                # 由于类别3，6作为未知类与类别15，16做了交换
                # 因此这里限制的是1, 4, 13, 6类别的样本数目
                if train_size > 20 and ((c == 1) or (c == 4) or (c == 13) or (c == 6)):
                    train_indices += random.sample(X, 10)  
                else:
                    train_indices += random.sample(X, train_size)
        else:
            if c > known_num:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            train_indices += random.sample(X, train_size)
    index = tuple(zip(*train_indices))
    train_gt[index] = gt[index]

    os_test_gt[index] = 0
    unknown_indices = np.nonzero(os_test_gt > known_num)
    os_test_gt[unknown_indices] = known_num + 1

    cs_test_gt = np.copy(os_test_gt)
    cs_test_gt[unknown_indices] = 0
    return train_gt, cs_test_gt, os_test_gt


def TrainTestSplit(datasetName, unknown_class):
    if datasetName == 'Indian' and (7 in unknown_class or 9 in unknown_class):
        raise Exception("Can't set the class 7 or 9 of Indian dataset to unknonw class.")
    
    info = DatasetInfo.info[datasetName]
    gt, class_num = load(datasetName) 
    unknown_num = len(unknown_class)
    known_num = class_num - unknown_num
    modify_labelID_by_knowID = gain_exchange_class(unknown_class, class_num)
    print(modify_labelID_by_knowID)
    gt_temp = np.copy(gt)
    for i, modefiyID_i in enumerate(modify_labelID_by_knowID[0]):
        modefiyID_index = np.nonzero(gt_temp==modefiyID_i)
        for index in zip(*modefiyID_index):
            x = index[0]
            y = index[1]
            gt[x,y]=modify_labelID_by_knowID[1][i]
    for i, modefiyID_i in enumerate(modify_labelID_by_knowID[1]):
        modefiyID_index = np.nonzero(gt_temp==modefiyID_i)
        for index in zip(*modefiyID_index):
            x = index[0]
            y = index[1]
            gt[x,y]=modify_labelID_by_knowID[0][i]
    
    H, W = gt.shape[0], gt.shape[1]
    for size in TRAIN_SIZE:
        print("data set:", datasetName, " label shape", gt.shape)
        print('class_num:', class_num)
        print('known_num:', known_num)
        print('total sample num', H * W)
        print('label sample num', np.sum(gt != 0))
        for r in range(RUN):
            train_gt, cs_test_gt, os_test_gt = sample_train_test(datasetName, gt, size, known_num, class_num)
            if r == 0:
                print("train set:", np.sum(train_gt != 0))
                print("test set:", np.sum(os_test_gt != 0))
                print("cs test set:", np.sum(cs_test_gt != 0))
                print("train class num: ", np.unique(train_gt))
                print("cs test class num: ", np.unique(cs_test_gt))
                print("os test class num: ", np.unique(os_test_gt))

            save_sample(train_gt, cs_test_gt, os_test_gt, datasetName, size, r)
    print('Finish split {}'.format(datasetName))


def gain_exchange_class(unknown_class, num_class):
    assert len(unknown_class) < num_class
    unknown_num = len(unknown_class)
    all_class = list(range(0, num_class + 1))
    tail_class = all_class[-unknown_num:]
    exchange_class1 = []
    exchange_class2 = []
    for class1 in unknown_class:
        if class1 not in tail_class:
            exchange_class1.append(class1)
    for class2 in tail_class:
        if class2 not in unknown_class:
            exchange_class2.append(class2)
    return [exchange_class1, exchange_class2]

if __name__ == '__main__':
    dir = './trainTestSplit/'
    dataseteNames = ['Indian', 'salinas', 'paviaU']
    unknown_class = {
                'Indian': [3, 6], 
                'salinas': [4, 12, 14],
                'paviaU':[9], 
    }
    if not os.path.isdir(dir):
        os.makedirs(dir)
    json_name = dir + "split_info.json"
    with open(json_name, 'w') as f:
        json.dump(unknown_class, f)
    for name in dataseteNames:
        TrainTestSplit(name, unknown_class[name])
    print('*'*8 + 'FINISH' + '*'*8)
