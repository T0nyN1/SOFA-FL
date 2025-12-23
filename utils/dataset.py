import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tabulate import tabulate


def get_dataset(type='mnist'):
    dataset = {'mnist': mnist_datasets, 'cifar10': cifar10_datasets, 'ciphar100': cifar100_datasets}
    classes = {'mnist': 10, 'cifar10': 10, 'cifar100': 100}
    sizes = {'mnist': (28, 28), 'cifar10': (32, 32), 'cifar100': (32, 32)}
    if type not in dataset.keys():
        raise ValueError('Invalid dataset type.')
    return dataset[type], classes[type], sizes[type] if type in sizes.keys() else None


def mnist_datasets(data_dir='dataset'):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set

def cifar10_datasets(data_dir='dataset'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),   # mean
                             (0.2470, 0.2435, 0.2616))   # std
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set

def cifar100_datasets(data_dir='dataset'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),   # mean
            (0.2675, 0.2565, 0.2761)    # std
        )
    ])
    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set

def partition_dataset(dataset, Y, n_classes, n_clients, alpha, seed, class_allocation=None):
    """
    Partitions a dataset into subsets for multiple clients, supporting both IID and non-IID cases.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to be partitioned.
        Y (np.array): The target labels of the dataset, used to group examples by class.
        n_classes (int): The number of unique classes in the dataset (e.g., 10 for MNIST).
        n_clients (int): The number of clients.
        alpha (float): The parameter controlling the distribution of data across clients.
                       If `alpha == -1`, the dataset is partitioned IID (Independent and Identically Distributed).
                       If `alpha > 0`, the dataset is partitioned non-IID using a Dirichlet distribution
        seed (int): torch random seed.

    Returns:
        List[torch.utils.data.Subset]: A list of `torch.utils.data.Subset` objects, where each subset represents the
                                       data assigned to a particular client.
    """
    clients = []

    # IID Case
    # TODO: class_allocation handling
    if alpha == -1:
        torch.manual_seed(seed)
        num_samples = len(dataset)
        num_samples_per_client = len(dataset) // n_clients
        indices = torch.randperm(num_samples)
        start_index = 0
        for _ in range(n_clients):
            clients.append(torch.utils.data.Subset(dataset, indices[start_index:start_index + num_samples_per_client]))
            start_index += num_samples_per_client

    # Non-IID Case
    else:
        np.random.seed(seed)
        indices = [np.where(Y == class_id)[0] for class_id in range(n_classes)]
        for i in range(len(indices)):
            np.random.shuffle(indices[i])
        if class_allocation is None:
            class_allocation = []
            for class_id in range(n_classes):
                allocation = np.random.dirichlet([alpha] * n_clients)
                class_allocation.append(allocation)

        client_indices = [[] for _ in range(n_clients)]
        for class_id in range(n_classes):
            allocation = class_allocation[class_id] * len(indices[class_id])
            allocation = allocation.astype(int)
            allocation[np.argmax(allocation)] += len(indices[class_id]) - allocation.sum()
            start_index = 0
            for i in range(n_clients):
                client_indices[i].extend(indices[class_id][start_index:start_index + allocation[i]])
                start_index += allocation[i]
        clients = [torch.utils.data.Subset(dataset, np.random.permutation(item)) for item in client_indices]

    distribution = _classes_distributions_info(clients, Y)
    return clients, class_allocation, distribution


def _classes_distributions_info(clients, Y):
    n_classes = len(np.unique(Y))
    headers = ["Client"] + [f"Class {i}" for i in range(n_classes)]
    table = []

    for cid, subset in enumerate(clients):
        labels = np.array(Y)[np.array(subset.indices, dtype=int)]
        counts = dict(zip(*np.unique(labels, return_counts=True)))
        full_counts = [int(counts.get(class_id, 0)) for class_id in range(n_classes)]
        table.append([f"Client {cid}"] + full_counts)

    return "\nClasses distributions:\n" + tabulate(table, headers=headers, tablefmt="outline")