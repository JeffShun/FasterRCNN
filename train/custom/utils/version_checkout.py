import tarfile
import argparse
import os
import pathlib
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)

# 指定备份的代码版本，解压覆盖

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='Renet50')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    version = pathlib.Path(args.version)
    tar_path = "./Logs" / version / "backup.tar"
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path='.')
    