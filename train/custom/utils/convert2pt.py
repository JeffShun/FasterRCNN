
import argparse
import os
import sys
import torch

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from config.model_config import network_cfg

def load_model(model_path):
    model = network_cfg.gen_network
    checkpoint = torch.load(model_path, map_location={"cuda:0":"cuda:0","cuda:1":"cuda:0","cuda:2":"cuda:0","cuda:3":"cuda:0"})
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/Renet50/150.pth')
    parser.add_argument('--output_path', type=str, default='./checkpoints/pt_model')
    args = parser.parse_args()
    return args

# branch
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['IS_SCRIPTING'] = '1'
    args = parse_args()
    model = load_model(args.model_path)
    model_jit = torch.jit.script(model)
    output_pt_dir = args.output_path
    os.makedirs(output_pt_dir, exist_ok=True)
    model_jit.save(os.path.join(output_pt_dir, 'model.pt'))

