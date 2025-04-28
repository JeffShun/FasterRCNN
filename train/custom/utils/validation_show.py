import os
import cv2
import numpy as np

def save_result_as_image(image, output, gt, classes_list, save_path, img_id):
    """
    显示预测结果和GT框，分别画图后拼接保存。
    """
    os.makedirs(save_path, exist_ok=True)

    color_map = [(200, 0, 0), (0, 128, 256), (0, 200, 0)]

    # 1. 图像预处理：Tensor -> Numpy (OpenCV格式)
    image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    image = (image - image.min()) / (image.max() - image.min() + 1e-5)  # 归一化到0-1
    image = (image * 255).astype(np.uint8)  # 转换为0-255的整数
    image = image[:,:,::-1]  # BGR -> RGB [H, W, C]
    pil_img_pred = image.copy()
    pil_img_gt = image.copy()

    # 2. 画预测框
    pred_boxes, pred_labels, pred_scores = output
    pred_boxes = pred_boxes.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy() if len(pred_labels) > 0 else []
    pred_scores = pred_scores.cpu().numpy() if len(pred_scores) > 0 else []

    if len(pred_boxes) > 0:
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box
            color = color_map[label % len(color_map)]
            name = classes_list[label]
            cv2.rectangle(pil_img_pred, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(pil_img_pred, f'{name}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. 画 GT 框
    gt_boxes = gt[0]  # [20, 5]
    valid_gt = gt_boxes[gt_boxes[:, 0] >= 0]  # 去除 padding

    if len(valid_gt) > 0:
        for gt_box in valid_gt:
            x1, y1, x2, y2, cls = gt_box.tolist()
            cls = int(cls)
            color = color_map[cls % len(color_map)]
            name = classes_list[cls]
            cv2.rectangle(pil_img_gt, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(pil_img_gt, f'{name}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

    # 4. 拼接图像：横向拼接 pred + gt
    concat_img = np.hstack((pil_img_pred, pil_img_gt))

    # 5. 保存图像
    save_name = os.path.join(save_path, f"{img_id+1}.jpg")
    cv2.imwrite(save_name, concat_img)

