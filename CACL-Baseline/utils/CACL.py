import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import ce_loss, masking, consistency_loss, infer, infer_allpixel
from utils.os_evaluate import getDistance, getClassCenterRepresentation, confuse_evaluate, evaluate
from utils.draw import draw_pred_figure
import math
from torch import nn
from sklearn.mixture import GaussianMixture


def train_prototype(model, optimizer, dataloaders, **kwargs):
    device = kwargs.get('device')
    model.train()
    model.to(device)
    train_losses = np.zeros(4)
    for step, (data_lb, data_ulb) in enumerate(zip(dataloaders['l_train'], dataloaders['ul_train'])):
        loss = train_step_prototype(model, optimizer, **data_lb, **data_ulb, **kwargs)
        train_losses += [loss['loss'], loss['sup_loss'], loss['ui_loss'], loss['cc_loss']]
        if step % 10 == 0:  # 每训练5次打印一次lr和本次迭代的损失
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('step:{} sup_loss:{:.3e} ui_loss:{:.3e} cc_loss:{:.3e} lr:{}'
                .format(step, loss['sup_loss'], loss['ui_loss'], loss['cc_loss'], lr))
    train_losses = train_losses / len(dataloaders['l_train'])
    train_loss = {"loss": float(train_losses[0]), "sup_loss": float(train_losses[1]),
                    "ui_loss": float(train_losses[2]), "cc_loss": float(train_losses[3])}
    return train_loss


def train_step_prototype(model, optimizer, x_lb, y_lb, x_ulb, **kwargs):
    optimizer.zero_grad()
    device = kwargs.get('device')
    known_num = kwargs.get('known_num')
    lambda_u = kwargs.get('lambda_u')
    lambda_cc = kwargs.get('lambda_cc')
    mask_ratio = kwargs.get("mask_ratio")
    x_lb, y_lb, x_ulb = x_lb.to(device), y_lb.to(device), x_ulb.to(device)
    
    num_lb, num_ulb = y_lb.shape[0], x_ulb.shape[0]# 有标注样本数量
    inputs = torch.cat((x_lb, x_ulb))  # 将有标注样本和无标注样本在批量维度连接
    feats, logits = model(inputs, with_feat=True)  # 前向传播
    logits_x_lb = logits[:num_lb]  # labeled样本的closed-set classifier预测 p
    logits_x_ulb_w = logits[num_lb:]
    logits_x_ulb_s = model.backbone.forward_mask(x_ulb, mask_ratio=mask_ratio)
    sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')  # closed-set Ls损失
    feat, label = feats[:num_lb], y_lb  #　训练样本特征与标签
    unlabel_feat = feats[num_lb:num_lb + num_ulb]  # 无标记样本特征

    with torch.no_grad(): ## 更新原型
        temp_prototype = getClassCenterRepresentation(feat, label, known_num)  # 计算当前批量原型
        prototype = model.update_prototype(temp_prototype, momentum=0.95)  # 更行原型
    # 计算标记样本和无标记样本与原型的距离
    label_dist = getDistance(feat, prototype)
    unlabel_dist = getDistance(unlabel_feat, prototype)

    with torch.no_grad(): ## 更新阈值
        class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean = get_dist_static_train(model, label_dist, label, known_num)  # 计算当批量阈值
        class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean = model.update_filter_thre(class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean, 0.9)
        ## 计算置信度掩码p_mask
        targets_p = F.softmax(logits_x_ulb_w, dim=-1)  # weak样本的closed-set预测概率分布
        p_mask = masking(targets_p, 0.95)
        ## 计算一致性正则化掩码，以及无标记样本原型对比indice
        pseudo_label = torch.argmax(logits_x_ulb_w, dim=-1)
        correspond_distance, reliable_idx, unreliable_idx = get_reliable_idx(unlabel_dist, pseudo_label, class_dist_mean, class_dist_std, interclass_dist_mean)
        ## CADT: Class-Adaptive Deviation Threshold (replaces fixed bias=2.0)
        kappa_c = model.get_cadt_kappa()
        filter_thre = class_dist_mean + kappa_c * class_dist_std
        filter_prototype_mask = (correspond_distance < filter_thre[pseudo_label])  # 用于过滤未知类的掩码
        reliable_idx = reliable_idx & p_mask
        mask = p_mask * filter_prototype_mask
        # Update pseudo counts for CADT in next iteration
        model.update_pseudo_counts(mask, pseudo_label)
    # print("usage:", (p_mask * filter_prototype_mask).sum().item() / len(unlabel_dist))
    # 计算原型过滤掩码filter_prototype_mask
    ui_loss = consistency_loss(logits_x_ulb_s, targets_p, 'ce', mask = mask)
    cc_loss = CCloss_soft(label_dist, label)
    
    total_loss = sup_loss + lambda_u * ui_loss + lambda_cc * cc_loss
    total_loss.backward()
    optimizer.step()
    filter_thre = filter_thre.mean()
    return {"loss": total_loss.item(), "sup_loss": sup_loss.item(), "ui_loss": ui_loss.item(),
            "cc_loss": cc_loss.item()}

def get_dist_static_train(model, distances, label, known_num):
    class_dist_mean = torch.zeros(known_num, device=distances.device, dtype=distances.dtype)
    class_dist_std = class_dist_mean.clone()
    interclass_dist_mean = class_dist_mean.clone()
    class_dist_max = class_dist_mean.clone()
    for i in torch.arange(0, known_num):
        is_class = (label == i)
        is_not_class = ~(is_class)
        inclass_dist = distances[is_class, i]
        is_not_class_dist = distances[is_not_class, i]
        if len(inclass_dist) > 0:
            class_dist_mean[i] = inclass_dist.mean()
            class_dist_max[i] = inclass_dist.max()
        if len(inclass_dist) > 0:
            class_dist_std[i] = (inclass_dist - model.class_dist_mean.data[i]).pow(2).mean().sqrt()
        if len(is_not_class_dist) > 0:  interclass_dist_mean[i] = is_not_class_dist.mean()
    return class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean

def get_reliable_idx(distance, pseudo_label, class_dist_mean, class_dist_std, interclass_dist_mean):
    correspond_distance = torch.gather(distance, -1, pseudo_label.unsqueeze(-1)).squeeze()
    filter_thre_bias2 = 0
    reliable_idx = correspond_distance < class_dist_mean[pseudo_label] + class_dist_std[pseudo_label] * filter_thre_bias2
    unreliable_idx = (distance > interclass_dist_mean).sum(dim=-1) == distance.shape[-1]
    return correspond_distance, reliable_idx, unreliable_idx

def unlabel_CCloss(min_unlabel_dist, dist_inclass_mean, dist_interclass_mean):
    inlier_dist = min_unlabel_dist[min_unlabel_dist < dist_inclass_mean]
    outlier_dist = min_unlabel_dist[min_unlabel_dist > dist_interclass_mean]
    # print(inlier_dist.shape[0], outlier_dist.shape[0])
    # print(inlier_dist.sum(), outlier_dist.sum())
    unlabel_cc_loss = (inlier_dist.sum() - outlier_dist.sum()) / min_unlabel_dist.shape[0]
    # print(unlabel_cc_loss)
    return unlabel_cc_loss

def unlabel_ccloss(unlabel_dist, reliable_idx, unreliable_idx, logits, **kwargs):
    ucc_loss = torch.tensor(0, device=unlabel_dist.device, dtype=unlabel_dist.dtype)
    if kwargs['epoch'] > 0:
        if reliable_idx.sum() > 0:
            cc_loss = CCloss_soft(unlabel_dist[reliable_idx], logits[reliable_idx])
            ucc_loss += cc_loss
        # if unreliable_idx.sum() > 0:
        #     sp_loss = -unlabel_dist[unreliable_idx].mean()
        #     ucc_loss += sp_loss
    return ucc_loss

def gaussian_f(x, a, b, c):
    return a * torch.exp(-(x - b).pow(2) / c)


def CCloss_soft(distance, label):
    close_extent = gaussian_f(distance, a=10, b=0, c=1)  # 使用高斯函数作为与原型的靠近程度
    loss_f = nn.CrossEntropyLoss(reduction='mean')
    cc_loss = loss_f(close_extent, label)
    return cc_loss


def get_dist_static(distances, label, known_num):
    class_dist_mean = torch.zeros(known_num, device=distances.device, dtype=distances.dtype)
    class_dist_std = class_dist_mean.clone()
    interclass_dist_mean = class_dist_mean.clone()
    class_dist_max = class_dist_mean.clone()
    for i in torch.arange(0, known_num):
        is_class = (label == i)
        is_not_class = ~(is_class)
        inclass_dist = distances[is_class, i]
        is_not_class_dist = distances[is_not_class, i]
        if len(inclass_dist) > 0:
            class_dist_mean[i] = inclass_dist.mean()
            class_dist_max[i] = inclass_dist.max()
        if len(inclass_dist) > 0:
            class_dist_std[i] = (inclass_dist - class_dist_mean[i]).pow(2).mean().sqrt()
        if len(is_not_class_dist) > 0:  interclass_dist_mean[i] = is_not_class_dist.mean()
    return class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean
    

def get_init_prototype(model, train_dataloader, use_vit=False, **kwargs):
    feats, _, targets = infer(model, train_dataloader, use_vit, **kwargs)
    prototype = getClassCenterRepresentation(feats, targets, kwargs["known_num"])  # 构建类别原型
    label_dist = getDistance(feats, prototype)
    class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean = get_dist_static(label_dist, targets, kwargs["known_num"])
    return prototype, class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean


def test_overall_allpixel_thres(model, train_loader, test_loader, fig_path, **kwargs):
    device, known_num = kwargs.get('device'), kwargs.get('known_num')
    model.eval()
    model.to(device)
    class_center = model.prototype.data
    
    ## 训练数据前向推理，并计算出阈值
    train_feats, _, train_targets = infer(model, train_loader, **kwargs)
    threshold_li = get_class2class_gaussian_threshold(train_feats, train_targets, class_center, **kwargs)
    
    ## 测试数据前向推理
    feats, logits, targets, all_feats, all_logits = infer_allpixel(model, test_loader, **kwargs)
    preds = torch.argmax(logits, dim=-1)
    all_preds = torch.argmax(all_logits, dim=-1)
    distance = torch.cdist(feats, class_center, p=2) / math.sqrt(class_center.shape[-1])
    all_distance = torch.cdist(all_feats, class_center, p=2) / math.sqrt(class_center.shape[-1])   
    correspond_distance = torch.gather(distance, -1, preds.unsqueeze(-1)).squeeze()

    ## 计算close-set精度
    known_sample_indice = targets < known_num
    known_targets = targets[known_sample_indice]
    known_preds = preds[known_sample_indice]
    cs_acc_per_class, cs_OA, cs_AA, cs_kappa = evaluate(known_targets, known_preds, known_num)
    
    ## 不同类别使用不同阈值判断未知类
    threshold = threshold_li[preds]
    is_unknown = correspond_distance > threshold
    
    preds[is_unknown] = known_num
    acc_per_class, unknown_acc, OA, kappa, AA, confuse_matrix = confuse_evaluate(targets, preds, known_num + 1)

    ## 画出整副HSI预测图
    all_correspond_distance = torch.gather(all_distance, -1, all_preds.unsqueeze(-1)).squeeze()
    threshold = threshold_li[all_preds]
    is_unknown = all_correspond_distance > threshold
    all_preds[is_unknown] = known_num
    draw_pred_figure(all_preds, test_loader.dataset.h, test_loader.dataset.w, fig_path, kwargs['datasetName'], known_num)

    return dict(cs_acc_per_class=cs_acc_per_class, cs_OA=cs_OA, cs_AA=cs_AA, cs_kappa=cs_kappa,
                acc_per_class=acc_per_class, OA=OA, AA=AA, kappa=kappa, unknown_acc=unknown_acc,
                ), confuse_matrix


def gauss_cross_point_threshold(mu1, sigma1, mu2, sigma2):
    a = sigma1 ** 2 - sigma2 ** 2
    b = 2 * (sigma2 ** 2 * mu1 - sigma1 ** 2 * mu2)
    c = sigma1 ** 2 * mu2 ** 2 - sigma2 ** 2 * mu1 ** 2 - 2 *sigma1 ** 2 * sigma2 ** 2 * np.log(sigma1 / sigma2)
    root = np.poly1d([a, b, c]).r
    if sigma1 <= sigma2:
        return root.max()
    else:
        return root.min()


def gauss_fit(X):  # 单构成高斯混合模型
    # 输入数据分布X
    gmm = GaussianMixture(n_components=1, covariance_type='diag')
    gmm.fit(X.reshape(-1, 1))
    # 返回高斯模型拟合的数据分布的均值mu和标准差sigma
    return np.squeeze(gmm.means_).item(), np.sqrt(np.squeeze(gmm.covariances_)).item()


def get_class2class_gaussian_threshold(train_feats, train_targets, class_center, **kwargs):
    distances = torch.cdist(train_feats, class_center, p=2) / math.sqrt(class_center.shape[-1])
    threshold_li = []
    for i in torch.arange(0, kwargs['known_num']):
        is_class = (train_targets == i)
        is_not_class = ~(is_class)
        inclass_dist = distances[is_class, i].cpu().numpy()
        is_not_class_dist = distances[is_not_class, i].cpu().numpy()
        mu1, sigma1 = gauss_fit(inclass_dist)
        mu2, sigma2 = gauss_fit(is_not_class_dist)
        threshold = gauss_cross_point_threshold(mu1, sigma1, mu2, sigma2)
        threshold_li.append(threshold)
    threshold_li = torch.tensor(threshold_li, device=train_feats.device, dtype=train_feats.dtype)
    return threshold_li
