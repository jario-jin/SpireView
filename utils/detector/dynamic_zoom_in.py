# Copyright (c) Spire, Inc. All Rights Reserved.
import numpy as np
import cv2
import os
import json
import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from spire_anno import find_contours, SpireAnno
from maskrcnn_detector import SpireDetector
from ag_map import agmap_total
from common_replay_buffer import ReplayBuffer


class ZoomInEnv(object):
    def __init__(self, image_list, gt_dir, agmap_dir, coarse_detector, fine_detector,
                 zoom_in_size=900, dataset='coco'):
        """
        :param image_list (list): 图像绝对路径list
        :param agmap_dir (str): 网络预测的agmap的路径
        :param coarse_detector (SpireDetector): 检测器，需要origin_size=False，min_image_size=(300,)
        :param fine_detector (SpireDetector): 检测器，需要origin_size=True
        """
        self.image_list = image_list
        self.gt_dir = gt_dir
        self.agmap_dir = agmap_dir
        self.spire_anno = SpireAnno(dataset=dataset)

        self.coarse_detector = coarse_detector
        self.fine_detector = fine_detector
        self.zoom_in_size = zoom_in_size

        self.n_images = len(image_list)
        print("Total images: {}".format(self.n_images))
        self.ids = np.random.permutation(self.n_images)
        self.cursor = -1
        # 0表示当前图像ZoomIn，1表示当前图像ZoomIn结束
        self.done = 1
        self.step_cnt = 0
        self.action_space = 9
    
    def step(self, action, n_actions=9, verbose=False):
        agmap = self.observation()
        assert n_actions == 9, 'only support n_actions=9 (palace)'

        sub_im, win, zoom_wh, wh = self._sub_window(action, np.sqrt(n_actions))
        prediction = self.fine_detector.detect(sub_im)
        prediction.bbox[:, 0] += win[0]
        prediction.bbox[:, 1] += win[1]
        prediction.bbox[:, 2] += win[0]
        prediction.bbox[:, 3] += win[1]
        prediction.size = zoom_wh
        prediction = prediction.resize(wh)
        TO_REMOVE = 1
        area = (prediction.bbox[:, 2] - prediction.bbox[:, 0] + TO_REMOVE) * \
               (prediction.bbox[:, 3] - prediction.bbox[:, 1] + TO_REMOVE)
        keep = area < 96 ** 2
        prediction = prediction[keep]

        scale = float(zoom_wh[0]) / wh[0]
        coarse_win = (int(round(win[0] / scale)), int(round(win[1] / scale)),
                      int(round(win[2] / scale)), int(round(win[3] / scale)))

        coarse_winbox = self._bbox_in_window(self.coarse_prediction, coarse_win)
        fine_winbox = self._bbox_in_window(prediction, coarse_win)
        gt_winbox = self._bbox_in_window(self.gt_boxlist, coarse_win)

        agval = agmap_total(gt_winbox, coarse_winbox, fine_winbox, reward=True, verbose=False)
        ag_c, ag_h, ag_w = self.agmap.shape
        scale_w, scale_h = float(zoom_wh[0]) / ag_w, float(zoom_wh[1]) / ag_h
        ag_win = (int(round(win[0] / scale_w)), int(round(win[1] / scale_h)),
                  int(round(win[2] / scale_w)), int(round(win[3] / scale_h)))
        self.agmap[:, ag_win[1]:ag_win[3], ag_win[0]:ag_win[2]] = 0

        if verbose:
            spire_anno = SpireAnno(dataset='coco')
            coares_show = spire_anno.visualize_boxlist(self.image, self.coarse_prediction, score_th=0.01)
            cv2.imshow('coarse', coares_show)
            fine_show = spire_anno.visualize_boxlist(self.image, prediction, score_th=0.01)
            cv2.imshow('fine', fine_show)

            zoom_im = cv2.resize(self.image, zoom_wh)
            cv2.rectangle(zoom_im, (win[0], win[1]), (win[2], win[3]), (0,210,0), 2)
            cv2.imshow('zoom_im', zoom_im)
            cv2.imshow('sub_im', sub_im)
            cv2.waitKey(100)
            print('pause?')

        self.step_cnt += 1
        if self.step_cnt >= 4:
            self.done = 1
            self.step_cnt = 0
        return self.agmap, agval, self.done

    def _bbox_in_window(self, boxlist, win):
        """
        返回窗口中的目标
        :param boxlist (BoxList): 输入所有的目标框
        :param win (tuple): 窗口
        :return: (BoxList)
        """
        x1, y1, x2, y2 = win
        w, h = boxlist.size
        if x1 <= w and x2 <= w and y1 <= h and y2 <= h:
            pass
        else:
            print("Window should in the image area, win:{}, size:{}".format(win, boxlist.size))

        assert boxlist.mode == 'xyxy', "Boxlist should in xyxy mode."
        keep_1 = boxlist.bbox[:, 0] >= x1
        keep_2 = boxlist.bbox[:, 0] <= x2
        keep_3 = boxlist.bbox[:, 1] >= y1
        keep_4 = boxlist.bbox[:, 1] <= y2
        keep_5 = boxlist.bbox[:, 2] >= x1
        keep_6 = boxlist.bbox[:, 2] <= x2
        keep_7 = boxlist.bbox[:, 3] >= y1
        keep_8 = boxlist.bbox[:, 3] <= y2
        keep = keep_1 & keep_2 & keep_3 & keep_4 & keep_5 & keep_6 & keep_7 & keep_8

        boxlist = boxlist[keep]
        boxlist.size = (x2 - x1, y2 - y1)
        return boxlist

    def _sub_window(self, action, split=3):
        min_size = self.zoom_in_size
        h, w, _ = self.image.shape
        if w > h:
            nh, nw = min_size, int(round(min_size * (float(w) / h)))
        else:
            nw, nh = min_size, int(round(min_size * (float(h) / w)))

        zoom_in_im = cv2.resize(self.image, (nw, nh))
        h_space = np.round(np.linspace(start=0, stop=nh, num=split + 1)).astype(np.int64)
        w_space = np.round(np.linspace(start=0, stop=nw, num=split + 1)).astype(np.int64)
        h_id = int(action // split)
        w_id = int(np.mod(action, split))
        x1, y1, x2, y2 = w_space[w_id], h_space[h_id], w_space[w_id + 1], h_space[h_id + 1]  # xyxy

        sub_im = zoom_in_im[y1:y2, x1:x2, :]
        # 裁剪图像，裁剪区域，放大后图像大小，原始图像大小
        return sub_im, (x1, y1, x2, y2), (nw, nh), (w, h)
    
    def observation(self):
        if self.done == 1:
            self.cursor += 1
            if self.cursor >= len(self.ids):  # 遍历数据集后，打乱序列，重新开始
                self.cursor = 0
                self.ids = np.random.permutation(self.n_images)

            self.image_name = os.path.basename(self.image_list[self.ids[self.cursor]])
            self.image = cv2.imread(self.image_list[self.ids[self.cursor]])
            self.coarse_prediction = self.coarse_detector.detect(self.image)

            gt_json = open(os.path.join(self.gt_dir, self.image_name + '.json'), 'r')
            gt_dict = json.loads(gt_json.read())
            self.gt_boxlist = self.spire_anno.to_boxlist(gt_dict)

            self.agmap = np.load(os.path.join(self.agmap_dir, self.image_name + '.npy'))
            self.done = 0
            self.observation_space = self.agmap.shape

        return self.agmap
    
    def reset(self):
        self.done = 1
        self.step_cnt = 0
        return self.observation()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZoomInQNet(nn.Module):
    """
    用于agmap的DQN网络
    """
    def __init__(self, in_channels, num_actions):
        super(ZoomInQNet, self).__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # 状态空间
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, self.num_actions)  # 动作空间

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def act(self, state, epsilon):
        """
        根据当前状态和epsilon选择动作
        :param state (np.ndarray): 当前状态
        :param epsilon (float): 选择网络作为输出的概率
        :return: (int64)
        """
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(dim=0).to(device)
            with torch.no_grad():
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()  # [0]-最大值，[1]-最大值的索引
        else:
            # 0-输出空间，随机整数
            action = random.randrange(self.num_actions)
        return action


def plot(frame_idx, rewards, losses, figure=False, hold=False):
    if figure:
        plt.figure(1)
        ax1 = plt.subplot(121)
        ax1.set_title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        ax1.plot(rewards, color='royalblue')
        ax2 = plt.subplot(122)
        ax2.set_title('loss')
        ax2.plot(losses, color='royalblue')
        if hold:
            plt.show()
        else:
            plt.pause(0.01)
    else:
        print('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))


num_frames = 100000
batch_size = 32
gamma = 0.9
replay_buffer = ReplayBuffer(10000)
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 5000


def compute_td_loss(model, optimizer, batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = model(state)
    with torch.no_grad():
        next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def dynamic_zoom_in_train(image_list, gt_dir, agmap_dir, config_file, weight_file):
    """
    :param image_list:
    :param gt_dir:
    :param agmap_dir:
    :param config_file:
    :param weight_file:
    :return:
    """
    # Epsilon贪婪搜索
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
                                         math.exp(-1. * frame_idx / epsilon_decay)

    coarse_detector = SpireDetector(config_file, weight_file, dataset='coco', min_image_size=(300,), origin_size=False)
    fine_detector = SpireDetector(config_file, weight_file, dataset='coco', origin_size=True)

    env = ZoomInEnv(image_list, gt_dir, agmap_dir, coarse_detector, fine_detector)
    state = env.observation()

    model = ZoomInQNet(env.observation_space[0], env.action_space)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    all_rewards = []
    episode_reward = 0
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(model, optimizer, batch_size)
            losses.append(loss.item())

        if frame_idx % 200 == 0:
            plot(frame_idx, all_rewards, losses)


if __name__ == '__main__':
    image_dir = '/home/jario/dataset/coco/minival2014'
    agmap_dir = '/media/jario/949AF0D79AF0B738/Experiments/fcos_predicted_agmap'
    gt_dir = '/home/jario/dataset/coco/minval2014_spire_annotation/annotations'
    image_fns = []
    fns = os.listdir(image_dir)
    fns.sort()

    config_file = '/home/jario/spire-net-1905/maskrcnn-benchmark-models/pre_trained/fcos_R_101_FPN_2x.yaml'
    weight_file = '/home/jario/spire-net-1905/maskrcnn-benchmark-models/pre_trained/FCOS_R_101_FPN_2x_s300.pth'


    image_list = [os.path.join(image_dir, f) for f in fns]
    dynamic_zoom_in_train(image_list, gt_dir, agmap_dir, config_file, weight_file)
