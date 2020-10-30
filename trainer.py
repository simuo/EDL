import torch
import torch.nn as nn
import torch.utils.data as data
import cfg
import os
from dataset import Mydataset
from darknet53 import MainNet
import matplotlib.pyplot as plt
from lookahead import Lookahead

device = cfg.DEVICE
savepath = 'models/net.pth'

conf_loss_fn = torch.nn.BCEWithLogitsLoss()  # 定义置信度损失函数
center_loss_fn = torch.nn.BCEWithLogitsLoss()  # 定义中心点损失函数
wh_loss_fn = torch.nn.MSELoss()  # 宽高损失
cls_loss_fn = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失


def loss_fn(output, target, alpha):
    """
    计算三个预测网络损失，返回有目标和无目标的损失和
    :param output: 网络输出
    :param target: 标签
    :param alpha: 看重的损失，放大某个损失
    :return: 网络总损失
    """
    output = output.permute(0, 2, 3, 1)  # 变换维度为N,C,H,W
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  #

    target = target.to(device)

    mask_obj = target[..., 4] > 0
    output_obj, target_obj = output[mask_obj], target[mask_obj]

    loss_obj_conf = conf_loss_fn(output_obj[:, 4], target_obj[:, 4])
    loss_obj_center = center_loss_fn(output_obj[:, 0:2], target_obj[:, 0:2])
    loss_obj_wh = wh_loss_fn(output_obj[:, 2:4], target_obj[:, 2:4])
    loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj[:, 5].long())
    loss_obj = loss_obj_conf + loss_obj_center + loss_obj_wh + loss_obj_cls

    # 负样本的时候只需要计算置信度损失
    mask_noobj = target[..., 4] == 0
    output_noobj, target_noobj = output[mask_noobj], target[mask_noobj]
    loss_noobj = conf_loss_fn(output_noobj[:, 4], target_noobj[:, 4])

    loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    return loss


if __name__ == '__main__':
    datas = Mydataset()
    imageDataloader = data.DataLoader(dataset=datas, batch_size=1, shuffle=True)

    net = MainNet(14).to(device)
    if os.path.exists(savepath):
        net.load_state_dict(torch.load(savepath))

    optim = torch.optim.Adam(net.parameters(), weight_decay=4e-4)

    base_opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))  # Any optimizer
    lookahead = Lookahead(base_opt, k=5, alpha=0.5)  # Initialize Lookahead

    losses = []
    epoch = 0
    while True:
        for i, (label13, label26, label52, img) in enumerate(imageDataloader):
            output_13, output_26, output_52 = net(img.to(device))

            loss_13 = loss_fn(output_13, label13, 0.9)
            loss_26 = loss_fn(output_26, label26, 0.9)
            loss_52 = loss_fn(output_52, label52, 0.9)

            loss = loss_13 + loss_26 + loss_52
            losses.append(loss)
            lookahead.zero_grad()
            loss.backward()  # Self-defined loss function
            lookahead.step()

            # plt.clf()
            # plt.plot(losses)
            # plt.pause(0.001)

        epoch += 1
        print('epoch-{}  |  loss:{}'.format(epoch, loss.item()))
        print('starting saveing!')
        torch.save(net.state_dict(), savepath)
        print('saved successfully!')
