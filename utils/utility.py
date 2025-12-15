import logging
import os
import re
from datetime import datetime

import torch


class PlainFormatter(logging.Formatter):
    ANSI_ESCAPE = re.compile(r'\033(?:[@-Z\\-_]|\[[0-9;]*[0-9A-Za-z])')

    def remove_ansi_colors(self, text):
        return self.ANSI_ESCAPE.sub('', text)

    def format(self, record):
        formatted_string = super().format(record)
        return self.remove_ansi_colors(formatted_string)


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

    plain_formatter = PlainFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(plain_formatter)
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


def union_find_groups(pairs):
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    groups = {}
    for x in parent:
        root = find(x)
        groups.setdefault(root, []).append(x)

    return list(groups.values())
