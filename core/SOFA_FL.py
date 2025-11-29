import copy
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.clustering import Hierarchical_Clustering, SHAPE
from core.managers import Train_Manager, Evaluation_Manager
from utils.dataset import partition_dataset


def train_client(model, train_loader, siblings, predecessor, optimizer, criterion, max_iters=None, device='cpu'):
    losses = []

    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        if max_iters is not None and i == max_iters:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, targets, model, siblings, predecessor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return np.mean(losses).item()


def eval_client(model, test_loader, device='cpu'):
    model.eval()
    total_err = 0
    total_counts = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        err = (targets != outputs.argmax(1)).sum()
        counts = torch.ones(len(inputs))

        total_err += err
        total_counts += counts.sum()

    accuracy = (1 - total_err / total_counts).item()
    return accuracy


def load_model(model, weights):
    return model.load_state_dict(torch.load(weights))


class SOFA_FL_Client:
    def __init__(self, id, model, train_loader, test_loader, optimizer, lr, weight_decay, local_updates, max_iters,
                 device='cpu'):
        self.id: int = id
        self.model: torch.nn.Module = copy.deepcopy(model).to(device) if model is not None else None
        self.train_loader = copy.deepcopy(train_loader)
        self.test_loader: DataLoader = copy.deepcopy(test_loader)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_func = optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=lr,
                                   weight_decay=weight_decay) if optimizer is not None else None
        self.local_updates = local_updates
        self.max_iters = max_iters
        self.device = device
        self.w = None

    def __repr__(self):
        return f"<SOFA_FL_Client(id={self.id}, w={self.w})>"

    def assign_optimizer(self):
        if self.optimizer is None:
            self.optimizer = self.optimizer_func(self.model.parameters(), lr=self.lr,
                                                 weight_decay=self.weight_decay) if self.optimizer_func is not None else None

    def local_update(self, criterion, siblings, predecessor):
        losses = []
        if self.local_updates is None:
            return
        for i in range(self.local_updates):
            loss = train_client(self.model, self.train_loader, siblings, predecessor,
                                optimizer=self.optimizer,
                                criterion=criterion,
                                max_iters=self.max_iters,
                                device=self.device)
            losses.append(loss)
        return self.id, losses

    def evaluate(self):
        acccuracy = eval_client(self.model, self.test_loader, self.device)
        return self.id, acccuracy

    def weights_flatten(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def samples(self):
        return len(self.train_loader.dataset) if isinstance(self.train_loader, DataLoader) else int(self.train_loader)


class SOFA_FL_Server:
    def __init__(self, n_clients, model, train_set, test_set, n_classes, optimizer, cfg, logger):
        self.n_clients = n_clients
        self.model = model
        self.optimizer = optimizer
        self.batch_size = cfg['client']['batch_size']
        self.comm_rounds = cfg['server']['comm_rounds']
        self.local_updates = cfg['client']['local_updates']
        self.max_iters = cfg['client']['max_iters']
        self.alpha = cfg['data']['alpha']
        self.seed = cfg['experiment']['seed']
        self.cfg = cfg
        self.device = torch.device(cfg['experiment']['device'])
        self.logger = logger

        Y = np.array(train_set.targets)
        if n_clients == 1:
            clients_train_loaders = [DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)]
            clients_test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)
            distribution_info = ""
        else:
            clients_datasets, distribution_info = partition_dataset(train_set, Y, n_classes, n_clients,
                                                                    alpha=self.alpha, seed=self.seed)
            clients_train_loaders = [DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True) for
                                     client_dataset
                                     in clients_datasets]
            clients_test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        self.logger.info("Dataset has been partitioned and assigned to clients." + distribution_info)

        if isinstance(self.local_updates, int):
            self.local_updates = [self.local_updates] * self.n_clients
        elif isinstance(self.local_updates, list):
            assert len(self.local_updates) == self.n_clients
        else:
            raise TypeError("Local-updates must be int or list")

        if self.max_iters is None:
            self.max_iters = [None] * self.n_clients
        elif isinstance(self.max_iters, int):
            self.max_iters = [self.max_iters] * self.n_clients
        elif isinstance(self.max_iters, list):
            assert len(self.max_iters) == self.n_clients
        else:
            raise TypeError("Max_iters must be int or list")
        self.clients = [
            SOFA_FL_Client(i, model,
                           train_loader,
                           clients_test_loader,
                           optimizer,
                           lr=self.cfg['client']['lr'],
                           weight_decay=self.cfg['client']['weight_decay'],
                           local_updates=local_updates,
                           max_iters=max_iters,
                           device=self.device)
            for i, (train_loader, local_updates, max_iters) in
            enumerate(zip(clients_train_loaders, self.local_updates, self.max_iters))]
        self.logger.info(f"Clients initialized. Number of clients: {len(self.clients)}")

        self.clients_dict = {}
        self.update_clients_dict()

        clients_weights_list = [client.weights_flatten() for client in self.clients]
        samples = [client.samples() for client in self.clients]
        self.architecture = Hierarchical_Clustering(clients_weights_list, samples,
                                                    type=self.cfg['server']['cluster_type'],
                                                    distance=self.cfg['server']['distance_metric'],
                                                    increment_factor=self.cfg['server']['increment_factor'],
                                                    device=self.device,
                                                    exp_dir=self.cfg['experiment']['experiment_dir'],
                                                    logger=self.logger)
        self.shape = SHAPE(tree=self.architecture)
        self.logger.info(f"Server initialized.")

    def __repr__(self):
        return "\n".join([c.__repr__() for c in sorted(self.clients, key=lambda c: c.index)])

    def update_clients_dict(self):
        self.clients_dict = {c.id: i for i, c in enumerate(self.clients)}

    def client_to_node(self, client: Union[SOFA_FL_Client, int]):
        if isinstance(client, int):
            client_id = client
        else:
            client_id = client.id
        if client_id not in self.architecture.node_dict.keys():
            return None
        index = self.architecture.node_dict[client_id]
        return self.architecture.nodes[index]

    def node_to_client(self, node: Union[SOFA_FL_Client, int]):
        if isinstance(node, int):
            node_index = node
        else:
            node_index = node.index
        if node_index not in self.clients_dict.keys():
            return None
        index = self.clients_dict[node_index]
        return self.clients[index]

    def aggregate_weights(self, clients):
        model = copy.deepcopy(clients[0].model)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = torch.sum(
                torch.stack([client.model.state_dict()[key].clone().to(self.device) * client.w for client in clients]),
                dim=0)
        model.load_state_dict(state_dict)
        return model

    def merge_weights(self, client_a, client_b):
        model = copy.deepcopy(client_a.model)
        state_dict = model.state_dict()
        w_a, w_b = client_a.w / (client_a.w + client_b.w), client_b.w / (client_a.w + client_b.w)

        for key in state_dict.keys():
            state_dict[key] = w_a * client_a.model.state_dict()[key] + w_b * client_b.model.state_dict()[key]
        model.load_state_dict(state_dict)
        return model

    def copy_weights(self, client):
        model = copy.deepcopy(client.model)
        return model

    def _update_ws(self, clients):
        samples = [client.samples() for client in clients]
        total_sampels = sum(samples)
        for client in clients:
            client.w = client.samples() / total_sampels

    def update_clients(self):
        if self.shape.log is None:
            new_nodes = self.architecture.nodes[len(self.clients):]
            for node in new_nodes:
                successors_indices = [n.index for n in node.successors]
                successor_clients = [self.clients[id] for id in successors_indices]
                self._update_ws(successor_clients)
                self.clients.append(
                    SOFA_FL_Client(node.index, self.aggregate_weights(successor_clients), node.samples,
                                   self.clients[0].test_loader, None, 0, 0, None, None, self.device))
            self.update_clients_dict()
        else:
            for node in self.shape.log.keys():
                weights = None
                if len(self.shape.log[node]) == 2:
                    node_a, node_b = self.shape.log[node]
                    client_a, client_b = self.node_to_client(node_a), self.node_to_client(node_b)
                    weights = self.merge_weights(client_a, client_b)
                    samples = client_a.samples() + client_b.samples()
                elif len(self.shape.log[node]) == 1:
                    client = self.node_to_client(self.shape.log[node][0])
                    weights = self.copy_weights(client)
                    # TODO: unprotected
                    samples = sum([n.samples for n in node.successors])

                if weights is not None:
                    self.clients.append(
                        SOFA_FL_Client(node.index, weights, samples, self.clients[0].test_loader, None, 0, 0, None,
                                       None, self.device))
                    self.update_clients_dict()

            for node in self.shape.log.keys():
                if len(self.shape.log[node]) == 0:
                    client = self.node_to_client(node)
                    if client is not None:
                        self.clients.remove(client)
                        self.update_clients_dict()

            for node in sorted(self.architecture.nodes, key=lambda n: n.level):
                if node.is_leaf():
                    continue
                successors = node.successors
                successors_clients = [self.node_to_client(s) for s in successors]
                self._update_ws(successors_clients)

    def sync_client_weights(self, client):
        node = self.client_to_node(client)
        node.centroid = client.weights_flatten()

    def train(self):
        train_manager = Train_Manager(server=self, logger=self.logger, val=True, visualize=True)
        train_manager.run()

    def eval(self):
        eval_manager = Evaluation_Manager(server=self, logger=self.logger)
        eval_manager.run()
