import argparse
import pprint

import tabulate
import yaml
from torchvision.models import resnet50, resnet34, resnet18

from backbone.ConvNet import ConvNet
from backbone.FCNet import FCNet
from core.SOFA_FL import SOFA_FL_Server
from utils.dataset import *
from utils.utility import *

tabulate.PRESERVE_WHITESPACE = True


def parse_args():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="SOFA-FL Federated Learning Framework")
    parser.add_argument('--config', type=str, default=os.path.join(BASE_DIR, 'configs/config.yaml'))
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--dataset_dir', type=str, default=os.path.join(BASE_DIR, 'dataset'))
    parser.add_argument('--experiment_dir', type=str, default=os.path.join(BASE_DIR, 'runs'))

    args = parser.parse_args()
    print("\n================= ⚙️ Experiment Configuration =================")
    pprint.pprint(vars(args), sort_dicts=False)
    print("================================================================\n")
    return args


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_model(type, in_channels, n_classes):
    if type == 'resnet50':
        model = resnet50(num_classes=n_classes)
    if type == 'resnet34':
        model = resnet34(num_classes=n_classes)
    if type == 'resnet18':
        model = resnet18(num_classes=n_classes)
    elif type == 'fcnet':
        model = FCNet(in_channels, n_classes)
    elif type == 'convnet':
        model = ConvNet()
    return model


def get_optimizer(type):
    optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'adamw': torch.optim.AdamW}
    if type not in optimizers.keys():
        raise ValueError('Invalid optimizer type')
    return optimizers[type]


def set_device(cfg):
    if cfg['experiment']['device'] == "auto" or cfg['experiment']['device'] is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        cfg['experiment']['device'] = device
    return cfg['experiment']['device']


def print_cfg(cfg, logger):
    table_data = []

    def flatten_to_strings(d, gap=0):
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                lines.extend(flatten_to_strings(v, gap=gap + 4))
            else:
                lines.append(f"{k}: {v}" if gap == 0 else f"{' ' * gap}{k}: {v}")
        return lines

    for i, section in enumerate(sections := list(cfg.keys())):
        params = cfg[section]
        if isinstance(params, dict):
            formatted_lines = flatten_to_strings(params)
            for idx, line in enumerate(formatted_lines):
                display_section = section if idx == 0 else ""
                table_data.append([display_section, line])
        else:
            table_data.append([section, str(params)])

        if i < len(sections) - 1:
            table_data.append([None, None])

    info = "Configs:\n" + tabulate.tabulate(table_data, headers=["Section", "Parameters"], tablefmt="outline",
                                            stralign="left")
    logger.info(info)


def main():
    args = parse_args()

    mode = args.mode
    experiment_dir = increment_dir(args.experiment_dir)
    logger = get_logger(mode, experiment_dir)
    cfg = load_config(args.config)
    cfg['experiment']['experiment_dir'] = experiment_dir
    device = set_device(cfg)
    print("Using device: ", device)
    print_cfg(cfg, logger)

    dataset_func, n_classes = get_dataset(cfg['train']['dataset'])
    if not os.path.exists(args.dataset_dir):
        os.mkdir(args.dataset_dir)
    train_set, test_set = dataset_func(args.dataset_dir)
    logger.info(
        f'Dataset loaded! Using dataset: {dataset_func.__name__} | n_classes: {n_classes} | n_train: {len(train_set)} | n_test: {len(test_set)}')

    model = get_model(cfg['train']['model'], cfg['data']['in_channels'], n_classes)

    if mode == 'train':
        optimizer = get_optimizer(cfg['train']['optimizer'])
        logger.info(f'Model initialized! Using model: {model.__class__.__name__} | optimizer: {optimizer.__name__}')

        server = SOFA_FL_Server(cfg['server']['n_clients'], model, train_set, test_set, n_classes, optimizer, cfg,
                                logger)
        server.train()
    elif mode == 'val':
        # TODO: val-only: load model weights then evaluate
        pass


if __name__ == '__main__':
    main()
