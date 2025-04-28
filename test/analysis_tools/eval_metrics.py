import os
import numpy as np
from glob import glob
from collections import defaultdict
import argparse

def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 加1 是为了对像素坐标做闭区间处理，可根据实际情况去掉
    inter_w = max(0, x2 - x1 + 1)
    inter_h = max(0, y2 - y1 + 1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def voc_ap(rec, prec):
    """按 VOC 2007 11 点插值法计算 AP"""
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # 使得 precision 单调递减
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 计算 recall 阶梯处的面积
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def evaluate_detection(npz_dir, class_names, iou_thresh=0.5):
    """
    读取 .npz 文件，计算每个类别的 Precision、Recall、F1、AP 以及 overall mAP。
    输入：
      npz_dir    - 存放 .npz 文件的目录，每个文件包含两个数组：
                     pred: [N,6] = [x1, y1, x2, y2, class_id, score]
                     label: [M,5] = [x1, y1, x2, y2, class_id]
      class_names - 类别名称列表，索引对应 class_id
      iou_thresh  - IoU 判定 TP/FP 的阈值，默认 0.5
    输出：
      per_class_ap, per_class_prec, per_class_rec, per_class_f1, mAP
    """

    # 用 dict 收集所有图片里每个类别的预测框和真实框
    # key: 类别索引，value: list of numpy arrays
    all_detections  = defaultdict(list)
    all_annotations = defaultdict(list)

    # 遍历目录下所有 .npz 文件
    for npz_path in glob(os.path.join(npz_dir, "*.npz")):
        data = np.load(npz_path)
        preds = data["pred"]   # shape [N,6]
        gts   = data["label"]  # shape [M,5]
        # 将每张图的预测和 GT 按类别拆分，存入对应列表
        for c in range(len(class_names)):
            all_detections[c].append(preds[preds[:, 4] == c])
            all_annotations[c].append(gts[gts[:, 4] == c])

    per_class_ap   = {}
    per_class_prec = {}
    per_class_rec  = {}
    per_class_f1   = {}

    # 对每个类别依次计算
    for c, cls_name in enumerate(class_names):
        # 拼接所有图片上这个类别的预测和 GT；若无则创建空数组
        dets = (np.concatenate(all_detections[c], axis=0)
                if len(all_detections[c]) > 0 else np.zeros((0, 6)))
        gts  = (np.concatenate(all_annotations[c], axis=0)
                if len(all_annotations[c]) > 0 else np.zeros((0, 5)))

        num_gt = len(gts)  # 该类别在所有图上的真实框总数

        # 如果根本没有预测或没有真实，所有指标置 0
        if dets.shape[0] == 0 or num_gt == 0:
            per_class_ap[c]   = 0.0
            per_class_prec[c] = 0.0
            per_class_rec[c]  = 0.0
            per_class_f1[c]   = 0.0
            continue

        # 1. 按置信度 score 降序排列预测框，越靠前越“有把握”
        dets = dets[np.argsort(-dets[:, 5])]

        # 准备标记 TP/FP 的数组，长度 = 预测框数
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        # matched 用来记录某个 GT 是否已被匹配，避免重复匹配
        matched = np.zeros(num_gt)

        # 2. 遍历每个预测框，判断是 TP 还是 FP
        for i, det in enumerate(dets):
            # 计算当前预测框与所有 GT 的 IoU
            ious = np.array([compute_iou(det[:4], gt[:4]) for gt in gts])
            if ious.size > 0:
                best_idx = ious.argmax()   # 找到 IoU 最大的 GT
                best_iou = ious[best_idx]
                # 如果 IoU 足够高且该 GT 尚未被匹配，则记为 TP
                if best_iou >= iou_thresh and matched[best_idx] == 0:
                    tp[i] = 1
                    matched[best_idx] = 1
                else:
                    # 否则记为 FP（要么 IoU 不够，要么 GT 已被匹配）
                    fp[i] = 1
            else:
                # 如果没有任何 GT，可视为 FP
                fp[i] = 1

        # 3. 累加 TP/FP，计算 Precision-Recall 曲线点
        acc_tp = np.cumsum(tp)    # 每个阈值位置的累计 TP
        acc_fp = np.cumsum(fp)    # 每个阈值位置的累计 FP
        recall    = acc_tp / (num_gt + 1e-6)                    # 召回率
        precision = acc_tp / (acc_tp + acc_fp + 1e-6)           # 精确率

        # 4. 在访问最后一个元素前，确保曲线非空
        if precision.size > 0 and recall.size > 0:
            ap    = voc_ap(recall, precision)                  # 计算 AP
            precv = precision[-1]                              # 最后一个阈值下的 Precision
            recv  = recall[-1]                                 # 最后一个阈值下的 Recall
            f1    = 2 * precv * recv / (precv + recv + 1e-6)   # F1 分数
        else:
            ap = precv = recv = f1 = 0.0

        # 保存结果
        per_class_ap[c]   = ap
        per_class_prec[c] = precv
        per_class_rec[c]  = recv
        per_class_f1[c]   = f1

    # 5. 最终 mAP 是所有类别 AP 的平均
    mAP = np.mean(list(per_class_ap.values()))

    return per_class_ap, per_class_prec, per_class_rec, per_class_f1, mAP

def print_results(class_names, ap, prec, rec, f1, mAP):
    header = f"{'Class':>10} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'AP':>7}"
    print("\nPer-class metrics:\n")
    print(header)
    print("-" * len(header))
    for c, name in enumerate(class_names):
        print(f"{name:>10} | "
              f"{prec[c]:7.4f} | "
              f"{rec[c]:7.4f} | "
              f"{f1[c]:7.4f} | "
              f"{ap[c]:7.4f}")
    print("-" * len(header))
    print(f"{'mAP@0.5':>10} = {mAP:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="目标检测评估：mAP, Precision, Recall, F1")
    parser.add_argument(
        "--npz_dir", type=str, default="data/output/FasterRCNN-FPN/raw_result")
    parser.add_argument(
        "--classes", nargs="+", default=["p1", "p2", "p3"])
    parser.add_argument(
        "--iou_thresh", type=float, default=0.5)
    args = parser.parse_args()

    ap, prec, rec, f1, mAP = evaluate_detection(
        args.npz_dir, args.classes, args.iou_thresh
    )
    print_results(args.classes, ap, prec, rec, f1, mAP)
