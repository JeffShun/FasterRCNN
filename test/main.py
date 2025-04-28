import argparse
import glob
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.font_manager as fm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import PredictModel, Predictor
from scipy.ndimage import zoom
import matplotlib.pylab as plt
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Test Object Detection')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='./data/input', type=str)
    parser.add_argument('--output_path', default='./data/output/FasterRCNN-FPN', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        default='../train/checkpoints/onnx_model/model.onnx'
        # default='../train/checkpoints/FasterRCNN-FPN/100.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def inference(predictor: Predictor, img: np.ndarray):
    pred_array = predictor.predict(img)
    return pred_array


def parse_label(classes_list, label_path):
    if os.path.exists(label_path):
        with open(label_path) as f:
            label_data = json.load(f)
            box_label = []
            for shape in label_data["shapes"]: 
                if shape["label"] in classes_list:
                    points = shape["points"]
                    x1, y1 = int(points[0][0]), int(points[0][1])
                    x2, y2 = int(points[1][0]), int(points[1][1])
                    label = shape["label"]
                    box_label.append((x1, y1, x2, y2, classes_list.index(label)))
        return np.array(box_label) 
    else:
        return None
    

def save_result(img, preds, label_map, save_path):
    color_map = [(200, 0, 0), (0, 128, 256), (0, 200, 0)]

    # 归一化图像并转换为 uint8
    img = img.astype("uint8")
    
    # 创建一个三通道图像（将灰度图像转为彩色）
    draw_img = np.zeros([img.shape[0], img.shape[1], 3])
    draw_img[:, :, 0] = img[:, :, 2]
    draw_img[:, :, 1] = img[:, :, 1]
    draw_img[:, :, 2] = img[:, :, 0]
    
    # 在图像上绘制框和文本
    for pred in preds:
        # 绘制矩形框
        cv2.rectangle(
            draw_img,
            (int(pred[0]), int(pred[1])),
            (int(pred[2]), int(pred[3])),
            color=color_map[int(pred[4])],
            thickness=2,
        )
        
        # 准备文本和字体
        text = "%s %.2f" % (label_map[int(pred[4])], pred[5])
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.5
        thickness = 2
        color = color_map[int(pred[4])]
        
        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 计算文本位置，确保文本不会超出图像边界
        text_x = int(pred[0])
        text_y = int(pred[1]) - 10
        
        # 如果文本超出图像左边界，则将其左移
        if text_x + text_width > draw_img.shape[1]:
            text_x = draw_img.shape[1] - text_width - 10
        
        # 如果文本超出图像上边界，则将其上移
        if text_y - text_height < 0:
            text_y = text_height + 10
        
        # 在图像上添加文本
        cv2.putText(
            draw_img,
            text,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
        )
    
    # 保存绘制后的图像
    cv2.imwrite(save_path, draw_img)

def save_result_with_gt(img, preds, gts, label_map, save_path):
    color_map = [(200, 0, 0), (0, 128, 256), (0, 200, 0)]
    img = img.astype("uint8")
    draw_img1 = np.zeros([img.shape[0], img.shape[1], 3])
    draw_img1[:, :, 0] = img[:, :, 2]
    draw_img1[:, :, 1] = img[:, :, 1]
    draw_img1[:, :, 2] = img[:, :, 0]
    draw_img2 = draw_img1.copy()

    for pred in preds:
        cv2.rectangle(
            draw_img1,
            (int(pred[0]), int(pred[1])),
            (int(pred[2]), int(pred[3])),
            color=color_map[int(pred[4])],
            thickness=2,
        )
        cv2.putText(
            draw_img1,
            "%s %.2f" % (label_map[int(pred[4])], pred[5]),
            (int(pred[0]), int(pred[1]) - 10),
            color=color_map[int(pred[4])],
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 0.5, 
            thickness = 2
        )

    for gt in gts:
        cv2.rectangle(
            draw_img2,
            (int(gt[0]), int(gt[1])),
            (int(gt[2]), int(gt[3])),
            color=color_map[int(gt[4])],
            thickness=2,
        )
        cv2.putText(
            draw_img2,
            "%s" % (label_map[gt[4]]),
            (int(gt[0]), int(gt[1]) - 10),
            color=color_map[int(gt[4])],
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 0.5, 
            thickness = 2
        )

    draw_img = np.concatenate((draw_img1,draw_img2), 1)
    cv2.imwrite(save_path, draw_img)


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    model_detection = PredictModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_detection = Predictor(
        device=device,
        model=model_detection,
    )
    os.makedirs(output_path, exist_ok=True)
    classes_list = ["p1","p2", "p3"]
    img_dir = os.path.join(input_path, "imgs")
    label_dir = os.path.join(input_path, "labels")

    for sample in tqdm(os.listdir(img_dir)):  
        img = np.array(Image.open(os.path.join(img_dir, sample)))
        gts = parse_label(classes_list, os.path.join(label_dir, sample.replace(".png",".json")))
        preds = predictor_detection.predict(img)
        if gts is not None:        
            save_result_with_gt(img, preds, gts, classes_list, os.path.join(output_path, sample))
            raw_result_dir = os.path.join(output_path, "raw_result")
            os.makedirs(raw_result_dir, exist_ok=True)
            # pred: [n, 6]  label:[n, 5]
            # print(preds.shape, gts.shape)
            np.savez_compressed(os.path.join(raw_result_dir, sample.replace(".png",".npz")), pred=preds, label=gts)
        else:
            save_result(img, preds, classes_list, os.path.join(output_path, sample))



if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )