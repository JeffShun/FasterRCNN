import numpy as np
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./train_data/origin_data/train/labels')
    parser.add_argument('--save_path', type=str, default='./train_data/origin_data/anchors.txt')
    parser.add_argument('--n_cluster', type=int, default=9)
    args = parser.parse_args()
    return args

def iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(boxes, k):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.random.random((rows,))
    np.random.seed()
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            cluster_menbers = boxes[nearest_clusters == cluster]
            # 加入异常判断，在有cluster成员时才计算
            if cluster_menbers.size > 0:
                clusters[cluster] = np.median(cluster_menbers, axis=0)
        last_clusters = nearest_clusters

    return clusters


def load_data(src_dir):
    data = []
    for sample in os.listdir(src_dir):
        data_path = os.path.join(src_dir, sample)
        with open(data_path) as f:
            label_data = json.load(f)
            for shape in label_data["shapes"]: 
                points = shape["points"]
                # 提取矩形框的左上角和右下角坐标
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                data.append([x2-x1,y2-y1])
    return np.array(data)


if __name__ == '__main__':
    args = parse_args()
    src_path = args.src_path
    save_path = args.save_path
    n_cluster = args.n_cluster
    data = load_data(src_path)
    # 使用k聚类算法
    out = kmeans(data,n_cluster) 
    out = out[np.argsort(out[:,0])]
    print(out)
    print('acc:{:.5f}%'.format(avg_iou(data,out) * 100))
    f = open(save_path, 'w')
    for i in range(out.shape[0]):
        if i == 0:
            x_y = "%d,%d" % (out[i][0], out[i][1])
        else:
            x_y = ", %d,%d" % (out[i][0], out[i][1])
        f.write(x_y)
    f.close()
