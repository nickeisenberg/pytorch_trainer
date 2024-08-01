import os
import re


def increment_save_root(save_root: str):
    if save_root.endswith(os.sep):
        save_root = save_root[:-len(os.sep)]

    if os.sep not in save_root:
        save_root = "./" + save_root

    base_dir = save_root.split(os.sep)[-1]
    root_dir = os.path.join(*save_root.split(os.sep)[:-1])
    
    if os.path.isdir(save_root):
        if re.search(r"_\d+$", base_dir):
            num = int(base_dir.split("_")[-1])
            base_dir = base_dir.split("_")[0] + f"_{num + 1}"
            save_root = os.path.join(root_dir, base_dir)
            increment_save_root(save_root)
        else:
            save_root = os.path.join(root_dir, base_dir + "_1")
            increment_save_root(save_root)

    else:
        os.makedirs(save_root)

