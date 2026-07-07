import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat
from utils.utils import weight_init, loadLabel, test, Log, seed_everything
from utils.CACL import train_prototype, get_init_prototype, test_overall_allpixel_thres
from Dataset.HSIDataset import LabelDataset, UnlabelDataset, DatasetInfo, LabelDataset_dynamic, AllPixelDataset
from Model.prototype import PAT_with_prototype
from Model.PAT import PAT
from Model.SGLPAT import SGLPAT
from torch.utils.data import DataLoader, sampler
import os
import argparse

SAMPLE_PER_CLASS = 10
RUN_NUM = 10
EPOCH_ITERS = 100
EPOCHS = 100


LR = 5e-4
BATCHSZ = 128
EVAL_BATCHSZ = 512
NUM_WORKERS = 0
SEED = 971105
Split_file_name = 'trainTestSplit'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_train_datalodaer(dataset, batchsz, num_workers):
    num_samplers = EPOCH_ITERS * batchsz
    dataloader = DataLoader(dataset,
                            batch_size=batchsz,
                            sampler=sampler.RandomSampler(dataset, replacement=True, num_samples=num_samplers),
                            num_workers=num_workers,
                            drop_last=False)
    return dataloader

def get_dataloader(data, label_train_gt, unlabel_train_gt, cs_test_gt, os_test_gt, patchsz, uratio):
    label_train_ds = LabelDataset(data, label_train_gt, patchsz=patchsz)
    unlabel_train_ds = UnlabelDataset(data, unlabel_train_gt, patchsz)
    cs_test_ds = LabelDataset_dynamic(data, cs_test_gt, patchsz=patchsz)
    all_test_ds = AllPixelDataset(data, os_test_gt, patchsz=patchsz)

    label_train_for_prototype = DataLoader(label_train_ds, batch_size=EVAL_BATCHSZ, drop_last=False, num_workers=NUM_WORKERS)
    label_train_dataloader = set_train_datalodaer(label_train_ds, BATCHSZ, NUM_WORKERS)
    unlabel_train_dataloader = set_train_datalodaer(unlabel_train_ds, BATCHSZ * uratio, 4 * NUM_WORKERS)
    cs_test_dataloader = DataLoader(cs_test_ds, batch_size=EVAL_BATCHSZ, drop_last=False, num_workers=4 * NUM_WORKERS)
    all_test_dataloader = DataLoader(all_test_ds, batch_size=EVAL_BATCHSZ, drop_last=False, num_workers=4 * NUM_WORKERS)

    return {"l_train_prototype": label_train_for_prototype,
            "l_train": label_train_dataloader,
            "ul_train": unlabel_train_dataloader,
            "cs_test": cs_test_dataloader,
            "all_test": all_test_dataloader}

def main(ROOT, MODEL_NAME, n_sample_per_class, run, **kwargs):
    datasetName = kwargs.get("datasetName")
    isExists = lambda path: os.path.exists(path)
    info = DatasetInfo.info[datasetName]
    data_path = "./data/{}/{}".format(datasetName, info['data_file_name'])
    isExists(data_path)
    data = loadmat(data_path)[info['data_key']]
    data = data.astype(np.float32)
    label_path = '{}/{}/sample{}_run{}.mat'.format(Split_file_name, datasetName, n_sample_per_class, run)
    isExists(label_path)
    train_gt, cs_test_gt, os_test_gt = loadLabel(label_path)
    train_gt, cs_test_gt, os_test_gt = train_gt.astype(int), cs_test_gt.astype(int), os_test_gt.astype(int)

    bands = data.shape[2]
    kwargs['known_num'] = known_num = int(np.max(train_gt))
    dataloaders = get_dataloader(data, train_gt, os_test_gt, cs_test_gt, os_test_gt, kwargs['patchsize'], kwargs['uratio'])

    use_sgl = kwargs.get('use_sgl', False)
    if use_sgl:
        backbone = SGLPAT(patchsz=kwargs['patchsize'], bands=bands, num_classes=known_num,
                    use_pos_embedding=kwargs['use_pos_embedding'],
                    use_pae_embedding=kwargs['use_pae_embedding'],
                    dis_type=kwargs['lambda_ucc'])
        model = PAT_with_prototype(backbone, known_num, kappa_base=2.0, gamma_cadt=0.5)
    else:
        backbone = PAT(patchsz=kwargs['patchsize'], bands=bands, num_classes=known_num,
                    use_pos_embedding=kwargs['use_pos_embedding'],
                    use_pae_embedding=kwargs['use_pae_embedding'],
                    dis_type=kwargs['lambda_ucc'])
        model = PAT_with_prototype(backbone, known_num)
    model.apply(weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc, best_epoch = 0, 0
    log = Log(ROOT, "result.txt")
    temp_trainLoss, temp_evalLoss = None, None
    prototype, class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean = get_init_prototype(model, dataloaders['l_train_prototype'], **kwargs)
    model.assign_prototype_and_filter_thre(prototype, class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean)
    for epoch in range(0, EPOCHS):
        kwargs['epoch'] = epoch
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        train_loss = train_prototype(model, optimizer, dataloaders, **kwargs)
        acc, evalLoss = test(model, criterion, dataloaders['cs_test'], use_vit=True, **kwargs)
        print('*' * 18)
        log.print('epoch:{} trainLoss:{:.3e} sup_loss:{:.3e} ui_loss:{:.3e} cc_loss:{:.3e} evalLoss:{:.8f} acc:{:.4f}'
              .format(epoch, train_loss["loss"], train_loss["sup_loss"], train_loss["ui_loss"],
                      train_loss["cc_loss"], evalLoss, acc))
        if acc > best_acc or epoch == EPOCHS - 1:
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                temp_trainLoss, temp_evalLoss = train_loss['loss'], evalLoss
            torch.save(model.state_dict(),
                       os.path.join(ROOT, '{}_sample{}_run{}_epoch{}.pkl'.format(MODEL_NAME, n_sample_per_class,
                                                                                 run, epoch)))
        print(ROOT)
        log.print('best acc:', best_acc, 'in epoch ', best_epoch,
              'with the train l:{:.3e} test l:{:.3e}'.format(train_loss['loss'], evalLoss))
    model.load_state_dict(
            torch.load(os.path.join(ROOT, '{}_sample{}_run{}_epoch{}.pkl'.format(MODEL_NAME, n_sample_per_class,
                                                                                 run, best_epoch))))
    result, confuse_matrix = test_overall_allpixel_thres(model, dataloaders['l_train_prototype'], dataloaders['all_test'], ROOT, **kwargs)
    log.print(result)
    log.print(confuse_matrix)
    return best_acc, best_epoch, temp_trainLoss, temp_evalLoss, result['OA'], result['unknown_acc']

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=300)
    parser = argparse.ArgumentParser(description='single train model')
    parser.add_argument('-d', '--dataset', type=str, default='Indian',
                        help='The name of dataset')
    parser.add_argument('-c', '--cuda', type=str, default='0',
                        help='the gpu id for training')
    parser.add_argument('--uratio', type=int, default=4)
    parser.add_argument('--lambda_u', type=float, default=1)
    parser.add_argument('--lambda_cc', type=float, default=1)
    parser.add_argument('--lambda_ucc', type=float, default=0)
    parser.add_argument('--use_pos_embedding', type=str2bool, default=False)
    parser.add_argument('--use_pae_embedding', type=str2bool, default=True)
    parser.add_argument('--mask_ratio', type=float, default=0.9)
    parser.add_argument('--patchsize', type=int, default=13)
    parser.add_argument('--use_sgl', type=str2bool, default=False,
                        help='Use SGL (Spectral Grouping Layer) instead of SPL')
    seed_everything(SEED)
    args = parser.parse_args()
    datasetName = args.dataset
    DEVICE = 'cuda:' + args.cuda
    
    all_os_accs = []
    all_unknown_accs = []

    for run in range(RUN_NUM):
        print(f"--- Starting run {run} ---")
        num = SAMPLE_PER_CLASS
        MODEL_NAME = "CACL_SGL" if args.use_sgl else "CACL"
        ROOT = 'check_point/{}/{}/{}/{}/{}'.format(Split_file_name, MODEL_NAME, datasetName, num, run)
        if not os.path.isdir(ROOT):
            os.makedirs(ROOT)
        best_acc, best_epoch, train_l, test_l, os_acc, unknown_acc = main(
            ROOT, MODEL_NAME, num, run, datasetName=datasetName, Split_file_name=Split_file_name, patchsize=args.patchsize,
            uratio=args.uratio, lambda_u=args.lambda_u,
            lambda_cc=args.lambda_cc, lambda_ucc=args.lambda_ucc, use_pos_embedding=args.use_pos_embedding,
            use_pae_embedding=args.use_pae_embedding, mask_ratio=args.mask_ratio,
            device=DEVICE, EPOCHS=EPOCHS, use_sgl=args.use_sgl
        )
        print("Best acc: {:.4f}, epoch: {}".format(best_acc, best_epoch))
        print("OS acc: {:.4f}, unknown acc: {:.4f}".format(os_acc, unknown_acc))
        all_os_accs.append(os_acc)
        all_unknown_accs.append(unknown_acc)

    # Calculate and print the average results
    avg_os_acc = sum(all_os_accs) / len(all_os_accs)
    avg_unknown_acc = sum(all_unknown_accs) / len(all_unknown_accs)
    final_log_path = 'check_point/{}/{}/{}/{}'.format(Split_file_name, MODEL_NAME, datasetName, SAMPLE_PER_CLASS)
    final_log = Log(final_log_path, "overall_results.txt")
    final_log.print("--- Overall Average Results ---")
    final_log.print("Average OS acc: {:.4f}".format(avg_os_acc))
    final_log.print("Average unknown acc: {:.4f}".format(avg_unknown_acc))
