import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from torchvision.transforms.functional import perspective as perspective_transform
from torchvision.transforms.functional import rotate
from torchvision.transforms.functional import InterpolationMode
from typing import Sequence, Optional, Union, Tuple


@MODEL_REGISTRY.register()
class RealESRNetModel(SRModel):
    """RealESRNet Model"""

    def __init__(self, opt):
        super(RealESRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt['queue_size']
        self.name = opt['name']
        if 'reg' in self.name:
            self.weight = float(opt['weight'])

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # training pair pool
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            print(self.queue_size, b)
            assert self.queue_size % b == 0, 'queue size should be divisible by batch size'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train:
            # 有空嵌入下面的代码
            rot90_prob= self.opt['rot90_prob'] # 旋转90概率
            undersample_prob= self.opt['undersample_prob'] # 采样概率
            center_fraction_range= self.opt['center_fraction_range'] # 中心分数范围
            acceleration_range= self.opt['acceleration_range'] # 加速度范围
            horizontal_mask_prob = self.opt['horizontal_mask_prob'] # 水平mask概率
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)
            # rotate 90
            if np.random.uniform(0, 1) < rot90_prob:
                self.gt = torch.rot90(self.gt, 1, [2, 3])
            expand = 1
            if 'reg' in self.name: expand = 2
            ori_h, ori_w = self.gt.size()[2:4]
            width, height = self.gt.size()[-1], self.gt.size()[-2]
            new_out = []
            for i in range(expand):
                # print(i)
                if i == 0:
                    self.kernel1 = data['kernel1'].to(self.device)
                    self.kernel2 = data['kernel2'].to(self.device)
                    self.sinc_kernel = data['sinc_kernel'].to(self.device)
                else:
                    self.kernel1 = data['kernel3'].to(self.device)
                    self.kernel2 = data['kernel4'].to(self.device)
                    self.sinc_kernel = data['sinc_kernel2'].to(self.device)
                # print(self.kernel1.shape, self.kernel2.shape, self.sinc_kernel.shape)
                # ----------------------- The first degradation process ----------------------- #

                def add_rician_noise(image, mean=0, std=0.05):
                    image = image.float()
                    # 生成高斯噪声并添加到图像上
                    noise_real  = torch.randn_like(image) * std + mean
                    noise_imaginary = torch.randn_like(image) * std + mean
                    noisy_image = torch.sqrt((image + noise_real)**2 + noise_imaginary**2)
                    return noisy_image

                def generate_random_mask(center_fractions: Sequence[float], accelerations: Sequence[int], num_cols: int, seed: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
                    if len(center_fractions) != len(accelerations):
                        raise ValueError("Number of center fractions should match number of accelerations")

                    rng = np.random.RandomState(seed)
                    choice = rng.randint(0, len(accelerations))
                    center_fraction = center_fractions[choice]
                    acceleration = accelerations[choice]

                    num_low_freqs = int(round(num_cols * center_fraction))
                    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)

                    mask = rng.uniform(size=num_cols) < prob
                    pad = (num_cols - num_low_freqs + 1) // 2
                    mask[pad: pad + num_low_freqs] = True

                    mask_shape = [1, 1] + [1] * (len(mask.shape) - 2)
                    mask_shape[-2] = num_cols
                    mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

                    # print("Generated Random Mask:")
                    # print(mask)
                    # print(f"Center Fraction: {center_fraction}, Acceleration: {acceleration}")
                    # true_count = int(mask.sum())
                    # print(f"Number of True values in the mask: {true_count}")

                    return mask

                def random_motion_transform(image, width, height,
                                rotate_prob = 0.5, rotate_range = [-5, 5],
                                translation_prob = [0.2, 0.2, 0.6], translation_range = [-0.1, 0.1],
                                perspective_prob = [0.3, 0.3, 0.4], perspective_range = [-0.1, 0.1],
                                stretch_prob = 0.4, stretch_range = [-0.1, 0.1]):
                    # 旋转图像 rotate
                    if np.random.uniform(0, 1) < rotate_prob:
                        rotate_angle = np.random.uniform(rotate_range[0], rotate_range[1])
                        out = rotate(image, rotate_angle, interpolation=InterpolationMode.BILINEAR)
                    else:
                        out = image

                    # 平移图像 translation
                    translation_type = np.random.choice(["left-right", "up-down", "keep"], p=translation_prob)
                    translation_rate = np.random.uniform(-translation_range[0], translation_range[0])
                    if translation_type == "left-right":
                        out = out.roll(int(translation_rate * width), 2)
                    elif translation_type == "up-down":
                        out = out.roll(int(translation_rate * height), 1)
                    else:
                        out = out

                    # 透视变换 perspective
                    original_pts = [[0, 0], [width, 0], [0, height], [width, height]]
                    perspective_type = np.random.choice(["pitch", "yaw", "keep"], p=perspective_prob)
                    pitch_rate1 = np.random.uniform(-perspective_range[0], perspective_range[0])
                    pitch_rate2 = np.random.uniform(-perspective_range[0], perspective_range[0])
                    yaw_rate1 = np.random.uniform(-perspective_range[1], perspective_range[1])
                    yaw_rate2 = np.random.uniform(-perspective_range[1], perspective_range[1])
                    if np.random.uniform(0, 1) < stretch_prob:
                        stretch_rate1 = np.random.uniform(stretch_range[0], stretch_range[1])
                        stretch_rate2 = np.random.uniform(stretch_range[0], stretch_range[1])
                    else:
                        stretch_rate1 = 0
                        stretch_rate2 = 0
                    if perspective_type == "pitch":
                        pitch_pts = [
                            [0 - pitch_rate1 * width, 0 - stretch_rate1 * height],
                            [width + pitch_rate1 * width, 0 - stretch_rate1 * height],
                            [0 - pitch_rate2 * width, height + stretch_rate2 * height],
                            [width + pitch_rate2 * width, height + stretch_rate2 * height],
                        ]
                        yaw_pts = original_pts
                        out = perspective_transform(out, original_pts, pitch_pts)
                    elif perspective_type == "yaw":
                        yaw_pts = [
                            [0 - stretch_rate1 * width, 0 - yaw_rate1 * height],
                            [width + stretch_rate2 * width, 0 - yaw_rate2 * height],
                            [0 - stretch_rate1 * width, height + yaw_rate1 * height],
                            [width + stretch_rate2 * width, height + yaw_rate2 * height],
                        ]
                        pitch_pts = original_pts
                        out = perspective_transform(out, original_pts, yaw_pts)
                    else:
                        pitch_pts = original_pts
                        yaw_pts = original_pts
                        stretch_pts = [
                            [0 , 0 - stretch_rate1 * height],
                            [width, 0 - stretch_rate1 * height],
                            [0 , height + stretch_rate2 * height],
                            [width, height + stretch_rate2 * height],
                        ]
                        out = perspective_transform(out, original_pts, stretch_pts)

                    return out

                def kspace_scan(image_tensor, K_data, cur_round, tol_round):
                    # 将图像张量转换为K空间数据
                    k_space_data = torch.fft.fft2(image_tensor, dim=(-2, -1))

                    # 进行 fftshift 操作将低频移到中心
                    k_space_data = torch.fft.fftshift(k_space_data, dim=(-2, -1))

                    # 获取图像的高度和宽度
                    _, H, W = image_tensor.shape

                    # 计算当前应该填充的行范围
                    start_row = cur_round * H // tol_round
                    end_row = (cur_round + 1) * H // tol_round

                    # 截取并填充到K_data中
                    K_data[:, start_row:end_row, :] = k_space_data[:, start_row:end_row, :]

                    return K_data

                # 转为单通道灰度图
                L_gt = self.gt.mean(dim=1, keepdim=False)

                rounds = np.random.choice(range(self.opt['rounds_range'][0], self.opt['rounds_range'][1] + 1, 2))

                K_data = np.zeros((L_gt.shape[0], L_gt.shape[1], L_gt.shape[2]), dtype=np.complex64)
                K_data = torch.from_numpy(K_data).to(self.device)

                for i in range(rounds):
                    if i > rounds * (5/11) and i < rounds * (6/11):
                        K_data = kspace_scan(L_gt, K_data, i, rounds)
                    else:
                        out_image = random_motion_transform(L_gt, width, height, rotate_prob=self.opt['rotate_prob'], rotate_range=self.opt['rotate_range'], translation_prob=self.opt['translation_prob'], translation_range=self.opt['translation_range'], perspective_prob=self.opt['perspective_prob'], perspective_range=self.opt['perspective_range'], stretch_prob=self.opt['stretch_prob'], stretch_range=self.opt['stretch_range'])

                        #out_image = center_crop(out_image, (400, 400))
                        K_data = kspace_scan(out_image, K_data, i, rounds)

                if np.random.uniform(0, 1) < self.opt['rician_noise_prob']:
                    temp_reconstructed_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(K_data, dim=(-2, -1)), dim=(-2, -1)))
                    rician_std = np.random.uniform(self.opt['rician_noise_range'][0], self.opt['rician_noise_range'][1])
                    temp_rician_image = add_rician_noise(temp_reconstructed_image, std=rician_std)
                    K_data = torch.fft.fft2(temp_rician_image, dim=(-2, -1))
                    K_data = torch.fft.fftshift(K_data, dim=(-2, -1))

                if np.random.uniform(0, 1) < undersample_prob:
                    # center_fraction = np.random.uniform(center_fraction_range[0], center_fraction_range[1])
                    acceleration = np.random.randint(acceleration_range[0], acceleration_range[1])
                    center_fraction = 4 / acceleration * 0.08
                    mask = generate_random_mask([center_fraction], [acceleration], K_data.shape[-1],)
                    # print(f"Center Fraction: {center_fraction}, Acceleration: {acceleration}", K_data.shape[-1])
                    mask = mask.to(self.device)
                    if np.random.uniform(0, 1) > horizontal_mask_prob:
                        mask = mask.t()
                    K_data = K_data * mask

                out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(K_data, dim=(-2, -1)), dim=(-2, -1)))
                #print(out.shape) # torch.Size([8, 400, 400])
                # 增加通道维度
                out = torch.unsqueeze(out, dim=1)
                #print(out.shape) # torch.Size([8, 1, 400, 400])
                # 增加通道数
                out = out.repeat(1, 3, 1, 1)
                #print(out.shape) # torch.Size([8, 3, 400, 400])
                new_out.append(out)
                #print('shape:', out.shape)

            out = torch.concat(new_out)
            #print(out.shape) # torch.Size([8, 3, 400, 400])
            self.gt = self.gt.repeat(expand, 1, 1, 1)
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            # training pair pool
            # self._dequeue_and_enqueue()
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

        '''
        import datetime
        import os
        import torchvision.transforms as transforms

        # Assuming self.lq and self.gt are PyTorch tensors with shape (batch_size, channels, height, width)
        batch_size = self.lq.size(0)

        # Get current time
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Create a folder to save images
        folder_path = f"/kaggle/working/images_{current_time}"
        os.makedirs(folder_path, exist_ok=True)

        for sample_index in range(batch_size):
            # Convert to PIL Image
            lq_image = transforms.ToPILImage()(self.lq[sample_index].cpu())
            gt_image = transforms.ToPILImage()(self.gt[sample_index].cpu())

            # Save image with current time and index as filename
            save_path = os.path.join(folder_path, f"lq_image_{current_time}_{sample_index}.png")
            save_path2 = os.path.join(folder_path, f"gt_image_{current_time}_{sample_index}.png")
            lq_image.save(save_path)
            gt_image.save(save_path2)

            print(f"Image saved at: {save_path}")
            print(f"Image saved at: {save_path2}")

        print(f"All images saved in folder: {folder_path}")
        '''

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if 'reg' in self.name:
            out = self.net_g.out
            #print('weight_shape:', self.weight)
            #print('original_out:', out.shape)
            out = torch.nn.functional.adaptive_avg_pool2d(self.net_g.out, 1)
            #print('pool_out:', out.shape)
            out = out.reshape(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
            #print('reshape_out:', out.shape)
            x1, x2 = torch.chunk(out, 2, 0)
            #print('x1:', x1.shape, 'x2:', x2.shape)
            x1 = x1 - x1.mean(0, keepdims=True)
            x2 = x2 - x2.mean(0, keepdims=True)
            l_reg, n_reg = torch.tensor(0).to(out), torch.tensor(0).to(out)
            for u, v in zip(x1, x2):
                #print('u:', u.shape, 'v:', v.shape)
                out_ = torch.cat([u, v])
                #print('cat_out:', out_.shape)
                nolinear_out = random_fourier_features_gpu(out_).reshape(out_.shape[0], -1)
                #print('nolinear_out:', nolinear_out.shape)
                a, b = torch.chunk(nolinear_out, 2, 0)
                #print('a:', a.shape, 'b:', b.shape)
                l_reg += self.weight * reg(u, v)
                n_reg += self.weight * reg(a, b)
            loss_dict['l_reg'] = l_reg
            loss_dict['n_reg'] = n_reg
            l_total += l_reg
            l_total += n_reg

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


def reg(x, y):
    cova_x = (x.t() @ x)
    cova_y = (y.t() @ y)

    mean_diff = (x - y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z


