
import os
import tarfile

# 训练时备份所有训练代码文件，以备后续查看
def model_backup(backup_path):
    with tarfile.open(backup_path, "w") as tar:
        for root, dirs, files in os.walk('./'):
            # 忽略 __pycache__ 文件夹
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            if "train_data" in dirs:
                dirs.remove("train_data")
        
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    tar.add(file_path, arcname=os.path.relpath(file_path, './'))

    