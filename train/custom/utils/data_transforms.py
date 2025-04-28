
import random
import torch
import numpy as np

"""
数据预处理工具
1、所有数据预处理函数都包含两个输入: img 、label
2、img、label的输入维度为3维[C,H,W]，第一个维度是通道数
"""

class TransformCompose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class to_tensor(object):
    def __call__(self, img, label):
        img_o = torch.from_numpy(img)
        label_o = torch.from_numpy(label)
        return img_o, label_o


class label_alignment(object):
    def __init__(self, max_box_num, pad_val=-1):
        self.max_box_num = max_box_num
        self.pad_val = pad_val

    def __call__(self, img, label):
        img_o, label_o = img.clone(), label.clone()
        pad_size = self.max_box_num - label.size(0)
        if pad_size > 0:
            pad_shape = (0, 0, 0, pad_size)
            label_o = torch.nn.functional.pad(label, pad_shape, value=self.pad_val)
        return img_o, label_o


class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        img_o, label_o = img.clone(), label.clone()
        _, ori_h, ori_w = img.shape
        scale_h = self.size[0] / ori_h
        scale_w = self.size[1] / ori_w
        img_o = torch.nn.functional.interpolate(img[None], size=self.size, mode="bilinear")[0]

        label_o[:, 0] *= scale_w
        label_o[:, 1] *= scale_h
        label_o[:, 2] *= scale_w
        label_o[:, 3] *= scale_h
    
        return img_o, label_o


class normlize(object):
    def __init__(self, win_clip=None):
        self.win_clip = win_clip

    def __call__(self, img, label): 
        img_o, label_o = img, label 
        if self.win_clip is not None:
            img = torch.clip(img, self.win_clip[0], self.win_clip[1])
        img_o = self._norm(img)
        return img_o, label_o
    
    def _norm(self, img):
        ori_shape = img.shape
        img_flatten = img.reshape(ori_shape[0], -1)
        img_min = img_flatten.min(dim=-1,keepdim=True)[0]
        img_max = img_flatten.max(dim=-1,keepdim=True)[0]
        img_norm = (img_flatten - img_min)/(img_max - img_min)
        img_norm = img_norm.reshape(ori_shape)
        return img_norm


class random_flip(object):
    def __init__(self, axis=1, prob=0.5):
        assert isinstance(axis, int) and axis in [1, 2]
        self.axis = axis
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img.clone(), label.clone()
        if random.random() < self.prob:
            _, height, width = img.shape
            img_o = torch.flip(img, [self.axis])
            if self.axis == 1:  # 对应于垂直翻转
                y1 = label[:, 1].clone()
                y2 = label[:, 3].clone()
                    
                label_o[:, 1] = height - 1 - y2
                label_o[:, 3] = height - 1 - y1

            if self.axis == 2:  # 对应于水平翻转
                x1 = label[:, 0].clone()
                x2 = label[:, 2].clone()

                label_o[:, 0] = width - 1 - x2
                label_o[:, 2] = width - 1 - x1
                
        return img_o, label_o



class random_rotate90(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img.clone(), label.clone()
        if random.random() < self.prob:
            direction = random.choice([1, -1])
            img_o = torch.rot90(img, direction, [1, 2])
            H, W = img.size(1), img.size(2)
            if direction == 1:
                # 顺时针旋转90度时，需要调整标注框的坐标
                x1 = label[:, 1]
                y1 = W - label[:, 2]
                x2 = label[:, 3]
                y2 = W - label[:, 0]
                label_o = torch.stack((x1, y1, x2, y2, label[:,4]), dim=1)
            if direction == -1:  
                # 逆时针旋转90度时，需要调整标注框的坐标
                x1 = H - label[:, 3]
                y1 = label[:, 0]
                x2 = H - label[:, 1]
                y2 = label[:, 2]
                label_o = torch.stack((x1, y1, x2, y2, label[:,4]), dim=1)
        return img_o, label_o



class random_gamma_transform(object):
    """
    input must be normlized before gamma transform
    """
    def __init__(self, gamma_range=[0.8, 1.2], prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob
    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_o = img**gamma
        return img_o, label_o
    

class random_add_gaussian_noise(object):
    def __init__(self, sigma_range=[0.1, 0.3], prob=0.5):
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img.clone(), label.clone()
        if random.random() < self.prob:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            noise = torch.randn_like(img) * sigma
            noisy_image = img + noise
            img_o = torch.clip(noisy_image, 0, 1)
        return img_o, label_o


class random_paste_objects(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img.clone(), label.clone()
        _, H, W = img.shape

        num_boxes = label.size(0)
        if random.random() > self.prob or num_boxes == 0:
            return img_o, label_o

        # 随机选择要粘贴的数量：1 ~ num_boxes
        paste_num = random.randint(1, num_boxes)
        selected_indices = random.sample(range(num_boxes), paste_num)

        pasted_labels = []
        used_boxes = label[:, :4].tolist()  # 原始框作为已用区域

        for idx in selected_indices:
            x1, y1, x2, y2, cls = label[idx]
            x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            patch_w, patch_h = x2 - x1, y2 - y1

            if patch_w <= 1 or patch_h <= 1:
                continue

            obj_patch = img[:, y1:y2, x1:x2].clone()

            for _ in range(10):  # 最多尝试10次不重叠粘贴
                new_x1 = random.randint(0, W - patch_w - 1)
                new_y1 = random.randint(0, H - patch_h - 1)
                new_x2 = new_x1 + patch_w
                new_y2 = new_y1 + patch_h
                new_box = [new_x1, new_y1, new_x2, new_y2]

                if not self._has_overlap(new_box, used_boxes):
                    # 粘贴
                    img_o[:, new_y1:new_y2, new_x1:new_x2] = obj_patch
                    used_boxes.append(new_box)
                    pasted_labels.append(torch.tensor([new_x1, new_y1, new_x2, new_y2, cls], dtype=torch.float32))
                    break  # 成功粘贴，跳出尝试

        if pasted_labels:
            label_o = torch.cat([label_o, torch.stack(pasted_labels)], dim=0)

        return img_o, label_o

    def _has_overlap(self, box, boxes, iou_thresh=0.0):
        for b in boxes:
            if self._iou(box, b) > iou_thresh:
                return True
        return False

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area








