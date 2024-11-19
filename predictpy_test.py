import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from skimage import io, exposure
from utils1 import data_pre
from collections import OrderedDict
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils1.utils import eval_all, accuracy
from thop import profile

from models.yynet_efficient_swin import CTCFNet as net
from torch.utils.data import DataLoader


class PredOption():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--pred_batch_size',required=False, default=8, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default='./', help='Location of the dataset used for testing ')
        parser.add_argument('--pred_dir', required=False, default='./', help='Location of the prediction results')

        parser.add_argument('--chkpt_path', required=False,
                            default='./',
                            help='checkpoints path')


        self.initialized = True
        return parser
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()
    def parser(self):
        self.opt = self.gather_options()
        return self.opt


def main():
    opt = PredOption().parser()

    # yynet
    model = net(pretrained=True)
    state_dict = model.transformer.state_dict()
    model.load_state_dict(torch.load(opt.chkpt_path), strict = False)
    model.cuda()
    model.eval()


    sample = torch.randn(1, 3, 256, 256).cuda()
    flops, _ = profile(model, inputs=(sample,))
    total = sum([param.nelement() for param in model.parameters()])
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(total / 1e6) + '{}'.format("M"))


    data_test = data_pre.Data(opt.test_dir)
    # data_test = data_pre.Data_test(opt.test_dir)
    test_loader = DataLoader(data_test, batch_size=opt.pred_batch_size)
    predict(model, data_test, test_loader, opt.pred_dir)


def predict(net, pred_set, pred_loader, pred_dir):
    pred_A_dir_rgb = pred_dir
    os.makedirs(pred_A_dir_rgb, exist_ok=True)

    acc_bank = []
    preds_all = []
    labels_all = []

    for i, (img_file_name, inputs, pack, masks, bounds) in enumerate(tqdm(pred_loader)):

        img_name, image, gt, mask, bound = img_file_name, inputs, pack, masks, bounds
        imgs = image.cuda().float()
        with torch.no_grad():
            preds,_,_,_,_ = net(imgs)
        preds = preds.cpu().detach()
        preds = nn.Softmax(dim=1)(preds)
        preds = preds.argmax(dim=1).long().numpy().squeeze()

        gt = gt.cuda().float()
        gt = (gt.detach().cpu().long()).numpy().squeeze()

        for (pred, gt) in zip(preds, gt):

            totall_acc = accuracy(pred, gt)
            acc_bank.append(totall_acc)
            preds_all.append(pred)
            labels_all.append(gt)

        for img_name,img in zip(img_name, preds):
            pred_save_dir = os.path.join(pred_A_dir_rgb, img_name)
            io.imsave(pred_save_dir, data_pre.Index2Color(img))
            print(pred_save_dir)

    F_score_use, Iou_mean_use, iou_use, F1_use, FBS = eval_all(preds_all, labels_all, num_class=6)
    print("FBS:{:.2f}".format(FBS * 100))
    print("MIoU:{:.2f}".format(Iou_mean_use * 100))
    print("F1:{:.2f}".format(F_score_use * 100))



if __name__ == '__main__':
    main()