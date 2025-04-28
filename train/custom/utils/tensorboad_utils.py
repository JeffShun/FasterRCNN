import os
import shutil
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from tensorboardX import SummaryWriter

def get_writer(tensorboad_dir):
    os.makedirs(tensorboad_dir, exist_ok=True)
    writer = SummaryWriter(tensorboad_dir)
    return writer

def display_img(imgs):
    org_img, target_img, pred_img = imgs
    fig = plt.figure(figsize=(12,5))
    shape = (1, 3)
    ax1 = plt.subplot2grid(shape, loc=(0,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid(shape, loc=(0,1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid(shape, loc=(0,2), rowspan=1, colspan=1)
    ax1.imshow(org_img, cmap='gray')
    ax1.set_title('origin')
    ax2.imshow(target_img, cmap='gray')
    ax2.set_title('target')
    ax3.imshow(pred_img, cmap='gray')
    ax3.set_title('pred')
    return fig