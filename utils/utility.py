import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import logging

def get_logger(name, log_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def increment_dir(save_dir, name='run'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    targets = [0 if item == name else int(item.replace(name, "")) for item in os.listdir(save_dir) if name in item]
    num = '' if len(targets) == 0 else str(max(targets) + 1)
    des = os.path.join(save_dir, f'{name}{num}')
    os.makedirs(des, exist_ok=False)
    return des

def euclidean_distance(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    if y is None:
        return torch.cdist(x, x, p=2)
    return torch.cdist(x, y, p=2)

def manhattan_distance(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    if y is None:
        return torch.cdist(x, x, p=2)
    return torch.cdist(x, y, p=1)


