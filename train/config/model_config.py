import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import ResNet50_FPN
from custom.model.rpn import RPN
from custom.model.roi_head import ROIHead
from custom.model.network import FasterRCNN

class network_cfg:

    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'

    # normal
    img_size = (768, 1280)
    classes_list = ["p1","p2","p3"]
    
    # network
    fpn_fea_size = 256
    network = FasterRCNN(
        backbone = ResNet50_FPN(
            in_channel = 3, 
            block_name = "Bottleneck",
            fpn_fea_size = fpn_fea_size
            ),
        rpn = RPN(
            in_channels = fpn_fea_size, 
            scales = [10, 20, 30],
            aspect_ratios = [0.5, 1, 2],
            rpn_bg_threshold = 0.3,
            rpn_fg_threshold = 0.5,
            rpn_nms_threshold = 0.5,
            rpn_min_boxsize = 5,
            rpn_train_prenms_topk = 10000,
            rpn_test_prenms_topk = 1000,
            rpn_train_topk = 2000,
            rpn_test_topk = 400,
            rpn_loss_count = 256,
            rpn_pos_fraction = 0.1
            ),
        roi_head = ROIHead(
            in_channels = fpn_fea_size,
            num_classes = 4, # include background class
            spatial_scale = 1/8,
            roi_iou_threshold = 0.5,
            roi_low_bg_iou = 0.0,
            roi_nms_threshold = 0.01,
            roi_min_boxsize = 5,
            roi_topk_detections = 20,
            roi_score_threshold = 0.9,
            roi_pool_size = 7,
            fc_inner_dim = 1024,
            roi_loss_count = 128,
            roi_pos_fraction = 0.25
        )
    )

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            resize(img_size),
            random_gamma_transform(gamma_range=[0.8, 1.2], prob=0.5),
            random_flip(axis=1, prob=0.5),
            random_flip(axis=2, prob=0.5),
            random_rotate90(prob=0.2),
            random_add_gaussian_noise(sigma_range=[0.05, 0.1], prob=0.2),
            label_alignment(max_box_num=20, pad_val=-1)
            ])
        )
    
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            resize(img_size),
            label_alignment(max_box_num=20, pad_val=-1)
            ])
        )
    
    # train dataloader
    batchsize = 1
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 5e-4

    # scheduler
    milestones = [20,50,80]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 100
    valid_interval = 1
    checkpoint_save_interval = 1
    log_dir = work_dir + "/Logs/FasterRCNN-FPN"
    checkpoints_dir = work_dir + '/checkpoints/FasterRCNN-FPN'
    load_from = work_dir + '/checkpoints/FasterRCNN-FPN/50.pth'
