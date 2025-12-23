import logging
import os
import re
from datetime import datetime

import torch

ANSI_ESCAPE = re.compile(r'\033(?:[@-Z\\-_]|\[[0-9;]*[0-9A-Za-z])')


class PlainFormatter(logging.Formatter):
    def remove_ansi_colors(self, text):
        return ANSI_ESCAPE.sub('', text)

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
    console_handler.setLevel(logging.WARNING)
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

def implicit_fisher_distance(x: torch.Tensor, y: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    if y is None:
        y = x

    # 1. 维度扩展以利用广播机制
    # x: (N, 1, D)
    # y: (1, M, D)
    x_exp = x.unsqueeze(1)
    y_exp = y.unsqueeze(0)

    # 2. 计算隐式重要性权重 (基于参数量级)
    # 逻辑：(|w1| + |w2|) / 2。参数绝对值越大，权重越高。
    # 结果形状: (N, M, D)
    weights = (torch.abs(x_exp) + torch.abs(y_exp)) / 2.0

    # 归一化权重，防止 D 维度过大导致距离数值爆炸
    # 也可以不归一化，取决于你后续聚类的阈值设置
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + eps)

    # 3. 计算加权平方差
    # (x - y)^2 * weights
    # 结果形状: (N, M, D)
    diff_sq = (x_exp - y_exp) ** 2
    weighted_diff = diff_sq * weights

    # 4. 在参数维度 D 上求和，并开方得到欧式距离
    # 结果形状: (N, M)
    dist_matrix = torch.sqrt(torch.sum(weighted_diff, dim=-1) + eps)

    return dist_matrix

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


def is_required(a, b):
    if not b:
        return False
    return a == b or a in b
