import torch
import matplotlib.pyplot as plt
import skimage.io
from skimage import io
import numpy as np
import os
import cv2
from torchvision.transforms import functional as F
from models.yynet_efficient_swin import CLCFormer as net
import data_pre
from data_pre import normalize_image
from PIL import Image


def get_feature1(imgs):

    # configs
    save_dir = r'D:\filedata\data_BT\fjdata\feature_visualize_base'
    save_size = 256

    img = skimage.io.imread(imgs)
    img = normalize_image(img)
    img_name = imgs.split("\\")[-1].split('.')[0]
    img = F.to_tensor(img)
    img = img.unsqueeze(0).float()
    img = img.cuda()


    model_path = r"D:\filedata\data_BT\fjdata\checkpoints\base\sea2_eph_80_iou0.820387_f1_0.899946.pth"
    model = net(num_classes=6)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # extract_features = ['cnn1','cnn2','cnn3','cnn4','att1','att2','att3','att4','fuse1','fuse2','fuse3','fuse4']
    extract_features={}

    with torch.no_grad():
        pred = model(img)[0]
        all_dict = model.extract_features

    pred = pred.cpu().detach()
    pred = pred.sigmoid()
    pred = torch.argmax(pred, dim=1).long().numpy().squeeze()
    pred_savedir = os.path.join(save_dir, img_name + '.png')
    io.imsave(pred_savedir, data_pre.Index2Color(pred))


    for k, v in all_dict.items():
        feat = v[0].data
        feat_range = feat.shape[0]
        feats = feat[0,:,:]
        for i in range(1, feat_range):
            feats += feat[i,:,:]
        extract_features[k] = feats

    for k,v in extract_features.items():
        feature = v.cpu().numpy()
        feature_img = np.asarray(feature*255, dtype=np.uint8)
        feature_img[feature_img<0]=0
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        feature_save = os.path.join(save_dir, img_name)
        os.makedirs(feature_save, exist_ok=True)

        if feature_img.shape[0] < save_size:
            a = feature_img.shape[0]
            tmp_file = os.path.join(feature_save, str(k) + "_" + str(a) + '.png')
            tmp_img = feature_img.copy()
            tmp_img = cv2.resize(tmp_img, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
            tmp_img = Image.fromarray(tmp_img)
            tmp_img.save(tmp_file)

        feature_file = os.path.join(feature_save, str(k)+ '.png')
        cv2.imwrite(feature_file, feature_img)

def get_feature2(imgs):

    # configs
    save_dir = r"D:\filedata\data_BT\fjdata\feature_visualize"
    save_size = 256

    img = skimage.io.imread(imgs)
    img = normalize_image(img)
    img_name = imgs.split("\\")[-1].split('.')[0]
    save_dir = os.path.join(save_dir, img_name)
    os.makedirs(save_dir, exist_ok=True)
    img = F.to_tensor(img)
    img = img.unsqueeze(0).float()
    img = img.cuda()


    model_path = r"G:\filedata\BtSeg\lswj\yynet\sea2_eph_149_iou0.921011_f1_0.958527.pth"
    model = net(num_classes=6)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # extract_features = ['cnn1','cnn2','cnn3','cnn4','att1','att2','att3','att4','fuse1','fuse2','fuse3','fuse4']
    extract_features={}

    with torch.no_grad():
        pred = model(img)[0]
        all_dict = model.extract_features

    pred = pred.cpu().detach()
    pred = pred.sigmoid()
    pred = torch.argmax(pred, dim=1).long().numpy().squeeze()
    pred_savedir = os.path.join(save_dir, img_name + '.png')
    io.imsave(pred_savedir, data_pre.Index2Color(pred))


    for k, v in all_dict.items():
        feat = v[0].data
        channel_range = feat.shape[0]
        for i in range(channel_range):
            feature = feat.cpu().numpy()
            feature_img = feature[i,:,:]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            feature_img = 255 - feature_img
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            feature_img1 = Image.fromarray(feature_img)
            save_path1 = os.path.join(save_dir, k)
            os.makedirs(save_path1, exist_ok=True)

            if feature_img.shape[0] < save_size:
                a = feature_img.shape[0]
                tmp_img = feature_img.copy()
                tmp_file = os.path.join(save_path1, str(k) + "_" + str(i) + "_" + str(a) + '.png')
                tmp_img = cv2.resize(tmp_img, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
                tmp_img = Image.fromarray(tmp_img)
                tmp_img.save(tmp_file)

            feature_file = os.path.join(save_path1, str(k) + "_" + str(i) + '.png')
            feature_img1.save(feature_file)

def get_feature3(imgs):

    # configs
    save_dir = r"D:\filedata\data_BT\fjdata\feature_visualize2"
    save_size = 256

    img = skimage.io.imread(imgs)
    img = normalize_image(img)
    img_name = imgs.split("\\")[-1].split('.')[0]
    save_dir = os.path.join(save_dir, img_name)
    os.makedirs(save_dir, exist_ok=True)
    img = F.to_tensor(img)
    img = img.unsqueeze(0).float()
    img = img.cuda()


    model_path = r"D:\filedata\data_BT\fjdata\checkpoints\yynet\sea2_eph_149_iou0.921011_f1_0.958527.pth"
    model = net(num_classes=6)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # extract_features = ['cnn1','cnn2','cnn3','cnn4','att1','att2','att3','att4','fuse1','fuse2','fuse3','fuse4']
    extract_features={}

    with torch.no_grad():
        pred = model(img)[0]
        all_dict = model.extract_features

    pred = pred.cpu().detach()
    pred = pred.sigmoid()
    pred = torch.argmax(pred, dim=1).long().numpy().squeeze()
    pred_savedir = os.path.join(save_dir, img_name + '.png')
    io.imsave(pred_savedir, data_pre.Index2Color(pred))


    for k, v in all_dict.items():
        feat = v[0].data
        feature_img, channel = feat.max(dim=0)
        feature_img = feature_img.cpu().numpy()
        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        feature_img = 255 - feature_img
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        feature_img1 = Image.fromarray(feature_img)
        save_path1 = os.path.join(save_dir, str(k))
        os.makedirs(save_path1, exist_ok=True)

        if feature_img.shape[0] < save_size:
            a = feature_img.shape[0]
            tmp_img = feature_img.copy()
            tmp_file = os.path.join(save_path1, str(k)+ "_" + str(a) + '.png')
            tmp_img = cv2.resize(tmp_img, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
            tmp_img = Image.fromarray(tmp_img)
            tmp_img.save(tmp_file)

        feature_file = os.path.join(save_path1, str(k) + '.png')
        feature_img1.save(feature_file)


if __name__ == '__main__':
    img_dir = r"D:\filedata\data_BT\fjdata\data_test11_BUS\test\img_visualize"
    img_list = os.listdir(img_dir)
    for i in img_list:
        img_path = os.path.join(img_dir,i)
        get_feature2(img_path)



