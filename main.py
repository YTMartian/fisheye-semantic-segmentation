from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from collections import defaultdict
from torchvision import transforms
import torch.nn.functional as F
from loss import FocalLoss2d
import torch.optim as optim
from unet_model import UNet
import torch.nn as nn
from tqdm import tqdm
from glob import glob
import numpy as np
import random
import torch
import time
import copy
import cv2
import re
import os


class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, num_classes=2, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.transform = transform
        self.threshold = num_classes
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        #self.ids = self.ids[:10]
        self.trans = transforms.Compose([transforms.ToTensor(), ])
        print('Creating dataset with {} examples'.format(len(self.ids)))
        self.imgs = []
        self.masks = []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir + idx + '.png'
        img_file = self.imgs_dir + idx + '.png'
        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file, 0)
        if mask is None:
            mask = cv2.imread(mask_file.replace('leftImg8bit', 'gtFine_labelIds'), 0)
        mask[mask >= self.threshold] = 0
        img = img[:, :, (2, 1, 0)]

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
            mask = torch.from_numpy(mask)
        return img, mask



# 相似度损失
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def train(param, model, train_data, valid_data, num_classes=-1, ignore_index=19, continue_train=False):
    # 初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_dir = param['checkpoint_dir']

    train_size = train_data.__len__()
    valid_size = valid_data.__len__()
    # c, y, x = train_data.__getitem__(0)['trace'].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, drop_last=True)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    # metrics = defaultdict(float)
    class_weights = torch.tensor([8.6979065, 8.497886, 8.741297, 5.983605, 8.662319, 8.681756, 8.683093, 8.763641, 8.576978, 2.7114885]).cuda()

    # criterion = CrossEntropyLoss2d(weight=class_weights)
    criterion = FocalLoss2d(weight=None)

    start_epoch = 0
    best_loss = 1e50
    save_file = open(checkpoint_dir + 'result.txt', 'a')
    if continue_train:
        checkpoints = torch.load(checkpoint_dir + '/' + 'checkpoint-last.pth')
        model.load_state_dict(checkpoints['state_dict'])
        best_loss = checkpoints['best_loss']
        start_epoch = checkpoints['epoch'] + 1
        optimizer.load_state_dict(checkpoints['optimizer'])
        print('load last checkpoint.')
        print('best loss: {}.'.format(best_loss))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_mode = copy.deepcopy(model)
    for epoch in range(start_epoch, epochs):
        print('Epoch {}'.format(epoch))
        # 训练阶段
        model.train()
        train_loss_per_epoch = 0
        bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc='train: ')
        for batch_idx, (data, target) in bar:
            data, target = Variable(data.to(device)), Variable(target.to(device))
            data = data.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.LongTensor)
            #data = data.to(device, dtype=torch.float)
            #target = target.to(device, dtype=torch.long)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            # loss = calc_loss(pred, target, metrics)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss.item()
        # train_loss_per_epoch = metrics['loss'] / train_size
        # 验证阶段
        model.eval()
        # metrics = defaultdict(float)
        valid_loss_per_epoch = 0
        pix_num_or = np.zeros(num_classes, dtype=np.float)
        pix_num_and = np.zeros(num_classes, dtype=np.float)
        pix_num_TPFP = np.zeros(num_classes, dtype=np.float)
        pix_num_TPFN = np.zeros(num_classes, dtype=np.float)
        with torch.no_grad():
            bar = tqdm(enumerate(valid_loader), total=len(valid_loader), ncols=100, desc='valid: ')
            for batch_idx, (data, target) in bar:
                # data, target = Variable(data.to(device)), Variable(target.to(device))
                # data = data.type(torch.cuda.FloatTensor)
                # target = target.type(torch.cuda.LongTensor)
                data = data.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.long)
                predict = model(data)
                loss = criterion(predict, target)
                valid_loss_per_epoch += loss.item()
                # loss = calc_loss(predict, target, metrics)

                predict = torch.argmax(predict, 1, keepdim=True)
                predict_onehot = torch.zeros(predict.size(0), num_classes, predict.size(2), predict.size(3)).cuda()
                predict_onehot = predict_onehot.scatter(1, predict, 1).float()
                target.unsqueeze_(1)
                target_onehot = torch.zeros(target.size(0), num_classes, target.size(2), target.size(3)).cuda()
                target_onehot = target_onehot.scatter(1, target, 1).float()

                # mask = (1 - target_onehot[:, ignore_index, :, :]).view(target.size(0), 1, target.size(2),
                #                                                        target.size(3))
                # predict_onehot *= mask
                # target_onehot *= mask

                area_and = predict_onehot * target_onehot
                area_or = predict_onehot + target_onehot - area_and

                pix_num_TPFP += torch.sum(predict_onehot,
                                          dim=(0, 2, 3)).cpu().numpy()
                pix_num_TPFN += torch.sum(target_onehot,
                                          dim=(0, 2, 3)).cpu().numpy()
                pix_num_and += torch.sum(area_and, dim=(0, 2, 3)).cpu().numpy()
                pix_num_or += torch.sum(area_or, dim=(0, 2, 3)).cpu().numpy()
        # valid_loss_per_epoch = metrics['loss'] / valid_size
        train_loss_per_epoch = train_loss_per_epoch / train_size
        valid_loss_per_epoch = valid_loss_per_epoch / valid_size
        precision_list = pix_num_and / (pix_num_TPFP + 1e-5)
        recall_list = pix_num_and / (pix_num_TPFN + 1e-5)
        iou_list = pix_num_and / (pix_num_or + 1e-5)

        # precision_list[ignore_index] = 0
        # recall_list[ignore_index] = 0
        # iou_list[ignore_index] = 0
        mean_precision = np.sum(precision_list) / (num_classes - 1)
        mean_recall = np.sum(recall_list) / (num_classes - 1)
        mean_iou = np.sum(iou_list) / (num_classes - 1)
        train_loss_total_epochs.append(train_loss_per_epoch)
        valid_loss_total_epochs.append(valid_loss_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存最优模型
        if valid_loss_per_epoch < best_loss:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'best_loss': best_loss, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = valid_loss_per_epoch
            best_mode = copy.deepcopy(model)
            print('best in epoch{}'.format(epoch))
        scheduler.step()
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'best_loss': best_loss, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-last.pth')
            torch.save(state, filename)
        # 显示loss
        if epoch % disp_inter == 0:
            print('Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}, MIoU:{}, MP:{}, MRC:{}\n'.format(
                epoch, train_loss_per_epoch,
                valid_loss_per_epoch,
                mean_iou, mean_precision, mean_recall))
        save_file.write(str(epoch) + ' ' + str(train_loss_per_epoch) + ' ' + str(valid_loss_per_epoch) + ' ' + str(
            mean_iou) + ' ' + str(mean_precision) + ' ' + str(mean_recall) + '\n')

    save_file.close()

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10

    root_dir = '../data/'
    train_imgs_dir = root_dir+'train/dst/'
    val_imgs_dir = root_dir+'val/dst/'
    train_labels_dir = root_dir+'train/anno_dst/'
    val_labels_dir = root_dir+'val/anno_dst/'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = RSCDataset(train_imgs_dir, train_labels_dir, num_classes=num_classes, transform=transform)
    valid_data = RSCDataset(val_imgs_dir, val_labels_dir, num_classes=num_classes)
    checkpoint_dir = 'model/'  # 模型保存路径
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    model = UNet(3, num_classes).to(device)

    # 参数设置
    param = {}
    param['epochs'] = 100  # 训练轮数
    param['batch_size'] = 4  # 批大小
    param['lr'] = 1e-3  # 学习率
    param['gamma'] = 0.9  # 学习率衰减系数
    param['step_size'] = 5  # 学习率衰减间隔
    param['momentum'] = 0.9  # 动量
    param['weight_decay'] = 0.  # 权重衰减
    param['checkpoint_dir'] = checkpoint_dir
    param['disp_inter'] = 1  # 显示间隔
    param['save_inter'] = 1  # 保存间隔
    # 训练
    train(param, model, train_data, valid_data, num_classes, continue_train=False)


if __name__ == '__main__':
    main()
