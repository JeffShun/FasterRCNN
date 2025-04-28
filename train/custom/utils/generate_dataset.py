"""生成模型输入数据."""

import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data')
    args = parser.parse_args()
    return args


def gen_lst(save_path, task, all_pids):
    save_file = os.path.join(save_path, task+'.txt')
    data_list = glob.glob(os.path.join(save_path, '*.npz'))
    num = 0
    with open(save_file, 'w') as f:
        for pid in all_pids:
            data = os.path.join(save_path, pid+".npz")
            if data in data_list:
                num+=1
                f.writelines(data.replace("\\","/") + '\n')
    print('num of data: ', num)


def get_max_box_count(data_path):
    max_box_count = 0
    for file_name in os.listdir(data_path):
        if file_name.endswith(".npz"):
            file_path = os.path.join(data_path, file_name)
            data = np.load(file_path, allow_pickle=True)
            label = data['label']
            max_box_count = max(label.shape[0],max_box_count) 
    print("max box num is {}".format(max_box_count))


def process_single(input):
    img_path, label_path, save_path, sample_id, classes_list = input
    img_arr = np.array(Image.open(img_path))
    with open(label_path) as f:
        label_data = json.load(f)
        box_label = []            
        for shape in label_data["shapes"]: 
            if shape["label"] in classes_list:
                points = shape["points"]
                # 提取矩形框的左上角和右下角坐标
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cls = classes_list.index(shape["label"])
                box_label.append((x1, y1, x2, y2, cls))

        # label_map = dict(zip(label_map_rev.values(), label_map_rev.keys()))
        # img_show = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        # for label in box_label:
        #     cv2.rectangle(
        #         img_show,
        #         (int(label[0]), int(label[1])),
        #         (int(label[2]), int(label[3])),
        #         color=(0, 200, 0),
        #         thickness=2,
        #     )
        #     cv2.putText(
        #         img_show,
        #         "%s" % (label_map[label[4]]),
        #         (int(label[0]), int(label[1]) - 10),
        #         color=(0, 200, 0),
        #         fontFace=cv2.FONT_HERSHEY_COMPLEX,
        #         fontScale=0.5,
        #     )

        # # 显示图像
        # cv2.imshow('Image with Boxes', img_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    if len(box_label) > 0:
        box_label = np.array(box_label)  
        np.savez_compressed(os.path.join(save_path, f'{sample_id}.npz'), img=img_arr, label=box_label)


if __name__ == '__main__':
    args = parse_args()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    classes_list = ["p1","p2", "p3"]

    for task in ["train","valid"]:
        print("\nBegin gen %s data!"%(task))
        img_dir = os.path.join(args.data_path, task, "imgs")
        label_dir = os.path.join(args.data_path, task, "labels")
        inputs = []
        all_ids = []
        for sample in tqdm(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, sample)
            sample_id = sample.replace('.png', '')
            label_path = os.path.join(label_dir, sample_id + ".json")
            inputs.append([img_path, label_path, save_path, sample_id, classes_list])
            all_ids.append(sample_id)
        pool = Pool(1)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(save_path, task, all_ids)

    get_max_box_count(save_path)

    