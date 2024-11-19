import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from models.yynet_efficient_swin import CTCFNet
from utils1.utils import *

import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import random
from optimezer_looka import Lookahead
from utils1 import data_pre
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime

def structure_loss(pred, mask):
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=6)
    wbce = ce_loss(pred, mask.squeeze(1).long())
    dice_loss = DiceLoss(6)
    dice = dice_loss(pred, mask, softmax=True)

    return wbce+dice


def accuracy(pred, label, ignore_zero = False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def train(train_loader, model, optimizer, epoch, best_iou, best_f1):
    model.train()
    # loss_record2, loss_record3, loss_record4,loss_record1 = AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter()
    # loss_bd, loss_bn = AvgMeter(), AvgMeter()
    loss_record3 = []
    loss_record2 = []
    loss_record1 = []
    loss_record = []
    acc_bank = []
    accum = 0
    model.cuda()
    txt_dir = opt.txt_dir
    for i, (img_file_name,inputs,pack,mask,bound) in enumerate(tqdm(train_loader)):        # 解包
        # ---- data prepare ----
        images, gts, masks, bounds = inputs, pack, mask, bound
        images = Variable(images).cuda().float()
        gts = Variable(gts).cuda().float()
        masks = Variable(masks).cuda().float()
        bounds = Variable(bounds).cuda().float()
        optimizer.zero_grad()
        # ---- forward ----ame and total
        map, bd2, bd1, bound2, bound1 = model(images)

        loss1 = structure_loss(map, gts)

        loss_bd2 = weighted_BCE_logits(bd2, masks)
        loss_bd1 = weighted_BCE_logits(bd1, masks)

        loss_bound2 = weighted_BCE_logits(bound2, bounds)
        loss_bound1 = weighted_BCE_logits(bound1, bounds)


        # loss weight
        loss_1 = loss1
        loss_2 = 0.6 * loss_bd1 + 0.4 * loss_bd2
        loss_3 = 0.6 * loss_bound1 + 0.4 * loss_bound2
        loss = 0.8*loss_1 + 0.10*loss_2 + 0.10*loss_3


        # ---- backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()

        # accauray
        res = map.sigmoid()
        pred = (torch.argmax(res, dim=1).data.cpu().long()).numpy().squeeze()
        gt = (gts.detach().cpu().long()).numpy().squeeze()
        acc_bank.append(accuracy(pred, gt))


        # ---- recording loss ----
        loss_record1.append(np.array(loss_1.item()))
        loss_record2.append(np.array(loss_2.item()))
        loss_record3.append(np.array(loss_3.item()))
        loss_record.append(np.array(loss.item()))


        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}],'
                  '[loss_category: {:.4f}, loss_mask: {:0.4f}, loss_bound: {:0.4f}]'.
                  format(epoch, opt.epoch, i, total_step,
                         np.nanmean(loss_record1), np.nanmean(loss_record2), np.nanmean(loss_record3)))

    write.add_scalar("train_toall_loss", np.nanmean(loss_record), epoch)
    write.add_scalar("train_category_loss", np.nanmean(loss_record1), epoch)
    write.add_scalar("train_mask_loss", np.nanmean(loss_record2), epoch)
    write.add_scalar("train_boundary_loss", np.nanmean(loss_record3), epoch)
    write.add_scalar("train_acc", np.nanmean(acc_bank), epoch)

    save_path = os.path.join(opt.train_save, opt.projectname,opt.data_name)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        F_score_use, meanIOU_use, iou_use, F1_use ,acc, FBS, loss_val = test(model, test_loader)

        with open(txt_dir, 'a') as file:
            time = datetime.datetime.now()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            file.write(save_path + "\n")
            file.write(time_str + "\n")
            file.write("epoch:{:2d}, miou:{:0.4f}, F1:{:0.4f}, FBS:{:.4f}, acc:{:.4f}, val_loss:{:.4f}".format(epoch, meanIOU_use,
                                                                                              F_score_use, FBS, acc, loss_val))
            file.write("\niou:[")
            for data_iou in iou_use:
                file.write('{:0.4f} '.format(data_iou))
            file.write(']\nF1:[')
            for f1 in F1_use:
                file.write('{:0.4f} '.format(f1))
            file.write(']\n-----\n\n')
            file.close()

        if (F_score_use + meanIOU_use) > (best_iou + best_f1):
            print('new best iou: ', meanIOU_use, F_score_use)
            best_iou = meanIOU_use
            best_f1 = F_score_use

            torch.save(model.state_dict(), os.path.join(save_path,
                                                        opt.data_name + 'eph_{:1d}_iou{:02f}_f1_{:02f}.pth'.format(
                                                            epoch, best_iou, best_f1)))
            print('[Saving Snapshot:]',
                  save_path + 'eph_%{:1d}_iou{:02f}_f1_{:02f}.pth'.format(epoch, best_iou, best_f1))

    write.add_scalar("val_toall_loss", loss_val, epoch)
    write.add_scalar("val_acc", acc, epoch)

    return best_iou


def test(model, test_data):

    model.eval()
    val_loss = AvgMeter

    loss_bank = []
    acc_bank = []
    preds_all = []
    labels_all = []



    for i, (img_file_name, inputs, pack, masks, bounds) in enumerate(tqdm(test_data)):

        image, gt, mask, bound = inputs, pack, masks, bounds
        image = image.cuda().float()
        gt = gt.cuda().float()
        masks = masks.cuda().float()



        with torch.no_grad():
            res, _, _, _, _ = model(image)
            # res = model(image)
        loss = structure_loss(res, gt)
        loss_bank.append(np.array(loss.item()))

        res = res.sigmoid()
        res = (torch.argmax(res, dim=1).data.cpu().long()).numpy().squeeze()
        gt = (gt.detach().cpu().long()).numpy().squeeze()


        for (pred, gt) in zip(res, gt):


            totall_acc = accuracy(pred, gt)
            acc_bank.append(totall_acc)
            preds_all.append(pred)
            labels_all.append(gt)


    F_score_use, Iou_mean_use, iou_use, F1_use, FBS = eval_all(preds_all, labels_all, num_class=6)
    # F_score_bd, Iou_mean_bd, iou_bd, F1_bd = eval_all(preds_bd, labels_bd, num_class=2)



    print('{} Loss: {:.4f}, F1_use: {:.4f},  IoU_use: {:.4f}, Acc: {:.4f}, FBS: {:.4f}'.
        format('test', np.mean(loss_bank), F_score_use, Iou_mean_use, np.mean(acc_bank), FBS))

    del preds_all, labels_all
    return F_score_use, Iou_mean_use, iou_use, F1_use, np.nanmean(acc_bank), FBS, np.nanmean(loss_bank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--projectname', type=str, default="project", help='model name')
    parser.add_argument('--data_name', type=str, default="dataset", help='data name')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--logs_path', type=str, default='./', help='path to logs')
    parser.add_argument('--txt_dir', type=str, default="./x.txt")
    parser.add_argument('--train_save', type=str, default='./checkpoints')

    parser.add_argument('--epoch', type=int, default=120, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    opt = parser.parse_args()
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # ---- build models ----
    model = CLCFormer(pretrained=True).cuda()

    # optimizer = Lookahead(base_optimizer)
    params = model.parameters()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    optimizer = torch.optim.AdamW(params, opt.lr, betas=(opt.beta1, opt.beta2))

    # logs
    write = SummaryWriter(os.path.join(opt.logs_path, opt.projectname, opt.data_name))

    # train data
    train_dataset = data_pre.Data('train', random_flip = True)
    val_dataset = data_pre.Data('test')
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, num_workers=2)
    test_loader = DataLoader(val_dataset, batch_size=opt.batchsize, num_workers=2)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    best_iou = 1e-5
    best_f1 = 1e-5
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epoch, eta_min=1e-6)

    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_iou, best_f1)
        scheduler.step()
        write.close()
