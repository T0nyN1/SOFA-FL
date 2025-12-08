import copy
import re
from itertools import zip_longest
from typing import Union, List, Dict

import numpy as np
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Subset, ConcatDataset

from core.clustering import Hierarchical_Clustering, SHAPE
from core.managers import Train_Manager, Evaluation_Manager
from utils.dataset import partition_dataset


def train_client(model, train_loader, siblings, predecessor, optimizer, criterion, max_iters=None, device='cpu'):
    losses = []
    local_losses = []

    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        if max_iters is not None and i == max_iters:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            local_loss, inter_loss, intra_loss = criterion(outputs, targets, model, siblings, predecessor)
            loss = local_loss + inter_loss + intra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        local_losses.append(local_loss.item())
    return np.mean(losses).item(), np.mean(local_losses).item()


def eval_client(model, test_loader, device='cpu'):
    model.eval()
    total_err = 0
    total_counts = 0
    for batch in test_loader:
        # TODO: fix it!
        if len(batch) == 3:
            inputs, targets = batch[0]
        elif len(batch) == 2:
            inputs, targets = batch

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
                 privacy=None, keep_data=True, device='cpu'):
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
        self.privacy: float = privacy
        self.keep_data: bool = keep_data
        self.device = device
        self.w = None

    def __repr__(self):
        return f"<SOFA_FL_Client(id={self.id}, w={self.w}, samples={self.samples()})>"

    def assign_optimizer(self):
        if self.optimizer is None:
            self.optimizer = self.optimizer_func(self.model.parameters(), lr=self.lr,
                                                 weight_decay=self.weight_decay) if self.optimizer_func is not None else None

    def local_update(self, criterion, siblings, predecessor):
        losses = []
        local_losses = []
        if self.local_updates is None:
            return
        for i in range(self.local_updates):
            loss, local_loss = train_client(self.model, self.train_loader, siblings, predecessor,
                                            optimizer=self.optimizer,
                                            criterion=criterion,
                                            max_iters=self.max_iters,
                                            device=self.device)
            losses.append(loss)
            local_losses.append(local_loss)
        return self.id, losses, local_losses

    def evaluate(self):
        acccuracy = eval_client(self.model, self.test_loader, self.device)
        return self.id, acccuracy

    def weights_flatten(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def samples(self, shareable=False):
        if not shareable:
            return len(self.train_loader.dataset) if self.train_loader is not None else 0
        else:
            data = self.get_shareable_data()
            return len(data.dataset) if data is not None else 0

    def initialize_share_loader(self):
        if self.privacy is not None and self.train_loader is not None:
            subset = Subset(self.train_loader.dataset,
                            np.random.choice(self.samples(), size=int(self.samples() * self.privacy),
                                             replace=False))
            self.share_loader = DataLoader(subset, batch_size=self.train_loader.batch_size, shuffle=True)

    def get_shareable_data(self):
        if hasattr(self, 'share_loader'):
            return self.share_loader
        else:
            return self.train_loader


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

        Y_train = np.array(train_set.targets)
        Y_test = np.array(test_set.targets)
        if n_clients == 1:
            # FIXME: variable reference.
            clients_train_loaders = [DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)]
            clients_test_loader = [DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)]
            train_distribution, test_distribution = "", ""
        else:
            clients_datasets, class_allocation, train_distribution = partition_dataset(
                train_set, Y_train,
                n_classes=n_classes,
                n_clients=n_clients,
                alpha=self.alpha,
                seed=self.seed
            )
            clients_test_datasets, _, test_distribution = partition_dataset(
                test_set, Y_test,
                n_classes=n_classes,
                n_clients=n_clients,
                alpha=self.alpha,
                seed=self.seed,
                class_allocation=class_allocation
            )
            clients_train_loaders = [DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True) for
                                     client_dataset in clients_datasets]
            # FIXME: DELETE after completion of implementation.
            # clients_test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
            clients_test_loaders = [DataLoader(d, batch_size=self.batch_size, shuffle=False) for d in
                                    clients_test_datasets]

        self.logger.info("Dataset has been partitioned and assigned to clients.")
        self.logger.info("Train " + train_distribution)
        self.logger.info("Test " + test_distribution)

        def _normalize_param(param_name):
            param = getattr(self, param_name.lower())
            if param is None:
                setattr(self, param_name.lower(), [None] * self.n_clients)
            elif isinstance(param, (int, float)):
                setattr(self, param_name.lower(), [param] * self.n_clients)
            elif isinstance(param, list):
                assert len(param) == self.n_clients
            else:
                raise TypeError(f"{param_name} must be a number or list")

        for param_name in ["Local_updates", "Max_iters"]:
            _normalize_param(param_name)

        self.clients = [
            SOFA_FL_Client(i, model,
                           clients_train_loaders[i],
                           clients_test_loaders[i],
                           optimizer,
                           lr=self.cfg['client']['lr'],
                           weight_decay=self.cfg['client']['weight_decay'],
                           local_updates=self.local_updates[i],
                           max_iters=self.max_iters[i],
                           privacy=None,
                           keep_data=False,
                           device=self.device)
            for i in range(self.n_clients)]
        self.logger.info(f"Clients initialized. Number of clients: {len(self.clients)}")

        self.clients_dict = {}
        self.update_clients_dict()

        clients_weights_list = [client.weights_flatten() for client in self.clients]
        samples = [client.samples() for client in self.clients]
        self.architecture = Hierarchical_Clustering(clients_weights_list, samples,
                                                    type=self.cfg['server']['cluster_type'],
                                                    distance=self.cfg['server']['distance_metric'],
                                                    threshold=self.cfg['server']['threshold'],
                                                    increment_factor=self.cfg['server']['increment_factor'],
                                                    device=self.device,
                                                    exp_dir=self.cfg['experiment']['experiment_dir'],
                                                    logger=self.logger)
        self.shape = SHAPE(tree=self.architecture,
                           graft_tolerance=self.cfg['server']['graft_tolerance'],
                           split_threshold=self.cfg['server']['split_threshold'],
                           merge_threshold=self.cfg['server']['merge_threshold'],
                           max_splits=self.cfg['server']['max_splits'],)

        self.data_share = cfg['client']['data_share']['activate']
        if self.data_share:
            self.share_ratios = cfg['client']["data_share"]['maximum_sharing_ratio']
            self.keep_data = cfg['client']["data_share"]['keep_data']
            for param in ["Share_ratios", "keep_data"]:
                _normalize_param(param)
            for i, client in enumerate(self.clients):
                client.privacy = self.share_ratios[i]
                client.keep_data = self.keep_data[i]

            self.data_share_manager = Data_Share_Manager(server=self,
                                                         mode=self.cfg['client']['data_share']['mode'],
                                                         gamma=self.cfg['client']['data_share']['gamma'])
            for client in self.clients:
                client.initialize_share_loader()

        self.logger.info(f"Server initialized.")

    def __repr__(self):
        return "<SOFA_FL_Server>\nClients:\n" + "\n".join(
            [c.__repr__() for c in sorted(self.clients, key=lambda c: c.id)])

    def update_clients_dict(self):
        self.clients_dict = {c.id: i for i, c in enumerate(self.clients)}

    def client_to_node(self, client: Union[SOFA_FL_Client, int]):
        if isinstance(client, int):
            client_id = client
        else:
            client_id = client.id
        if client_id not in self.architecture.nodes_dict.keys():
            return None
        index = self.architecture.nodes_dict[client_id]
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

    def merge_weights(self, clients):
        model = copy.deepcopy(clients[0].model)
        state_dict = model.state_dict()
        ws = [c.w for c in clients]
        w_sum = sum(ws)
        ws = [w / w_sum for w in ws]

        for key in state_dict.keys():
            state_dict[key] = sum([ws[i] * clients[i].model.state_dict()[key] for i in range(len(clients))])
        model.load_state_dict(state_dict)
        return model

    def copy_weights(self, client):
        model = copy.deepcopy(client.model)
        return model

    def _update_ws(self, clients: List[SOFA_FL_Client]):
        samples = [self.client_to_node(client).samples for client in clients]
        total_sampels = sum(samples)
        for i, client in enumerate(clients):
            client.w = samples[i] / total_sampels

    def _update_test_data(self, client: SOFA_FL_Client, successors: List[SOFA_FL_Client]):
        # TODO: use ConcatDataset
        test_loaders = []
        for s in successors:
            test_loader = s.test_loader
            if isinstance(test_loader, CombinedLoader):
                test_loaders.extend(test_loader._iterables)
            else:
                test_loaders.append(test_loader)
        combined_test_loader = CombinedLoader(test_loaders, mode="sequential")
        client.test_loader = combined_test_loader

    def update_clients(self):
        if self.shape.log is None:
            new_nodes = self.architecture.nodes[len(self.clients):]
            for node in new_nodes:
                successors_indices = [n.index for n in node.successors]
                successor_clients = [self.clients[id] for id in successors_indices]
                self._update_ws(successor_clients)
                self.clients.append(
                    SOFA_FL_Client(node.index,
                                   model=self.aggregate_weights(successor_clients),
                                   train_loader=None,
                                   test_loader=None,
                                   optimizer=None,
                                   lr=0,
                                   weight_decay=0,
                                   local_updates=None,
                                   max_iters=None,
                                   privacy=None,
                                   device=self.device))
            self.update_clients_dict()
        else:
            for node in self.shape.log.keys():
                weights = None
                if len(self.shape.log[node]) >= 2:
                    clients = [self.node_to_client(n) for n in self.shape.log[node]]
                    weights = self.merge_weights(clients)
                elif len(self.shape.log[node]) == 1:
                    client = self.node_to_client(self.shape.log[node][0])
                    weights = self.copy_weights(client)

                if weights is not None:
                    self.clients.append(
                        SOFA_FL_Client(node.index,
                                       model=weights,
                                       train_loader=None,
                                       test_loader=None,
                                       optimizer=None,
                                       lr=0,
                                       weight_decay=0,
                                       local_updates=None,
                                       max_iters=None,
                                       privacy=None,
                                       device=self.device))
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
                self._update_test_data(self.node_to_client(node), successors_clients)

    def sync_client_weights(self, client):
        node = self.client_to_node(client)
        node.centroid = client.weights_flatten()

    def train(self):
        train_manager = Train_Manager(server=self, logger=self.logger, val=self.cfg["experiment"]["evaluation"],
                                      visualize=self.cfg["output"]["mode"])
        train_manager.run()

    def eval(self):
        eval_manager = Evaluation_Manager(server=self, logger=self.logger)
        eval_manager.run()


class Data_Share_Manager():
    def __init__(self, server: SOFA_FL_Server, mode='adapt', gamma=5):
        self.server = server
        self.share_dict: Dict[SOFA_FL_Client: Dict[SOFA_FL_Client: float]] = None
        self.share_records = None
        self.gamma = gamma
        self.mode = mode

    def __repr__(self):
        return f"<Data_Share_Manager gamma={self.gamma}, mode={self.mode}>\nShare Records:\n{self.share_records_to_str()}"

    def share_records_to_str(self, col_width=15):
        ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

        def strip_ansi(s: str) -> str:
            return ANSI_RE.sub("", s)

        def pad(s: str, width: int) -> str:
            real_len = len(strip_ansi(s))
            if real_len >= width:
                return s
            return s + " " * (width - real_len)

        def color(s, active):
            RESET = "\033[0m"
            GRAY = "\033[90m"
            return f"{GRAY}{s}{RESET}" if not active else s

        cols = []

        for k, sub in self.share_records.items():
            if len(sub) > 0:
                active = k in self.server.clients_dict
                lines = [color(f"{k}:", active)]

                for sk, sv in sub.items():
                    lines.append(color(f"  {sk}: {sv}", active))

                cols.append(lines)

        rows = zip_longest(*cols, fillvalue="")
        output_lines = ["".join(pad(s, col_width) for s in row) for row in rows]

        return "\n".join(output_lines)

    def _calculate_s(self, distance):
        return self.gamma * torch.exp(-distance).item()

    def _id_to_key(self, id):
        for client in self.share_dict.keys():
            if client.id == id:
                return client
        return None

    def _validate_receiver_and_giver(self, receiver: SOFA_FL_Client, giver: SOFA_FL_Client):
        receiver_node = self.server.client_to_node(receiver)
        giver_node = self.server.client_to_node(giver)
        if receiver_node.is_leaf():
            return False
            # return self.server.architecture.find_predecessor(receiver_node) == self.server.architecture.find_predecessor(giver_node)
        else:
            if receiver.id in self.share_records and giver.id in self.share_records[receiver.id]:
                return False
            return giver_node is None or giver_node in receiver_node.get_leaves()

    def initialize(self):
        self.server.logger.info("Sharing weights initialized...")
        self.share_dict = {c: {} for c in self.server.clients}
        self.share_records = {c.id: {} for c in self.server.clients}
        shareable_clients_weights = torch.stack(
            [c.weights_flatten().to(self.server.device) for c in self.server.clients[:self.server.n_clients]])
        all_clients_weights = torch.stack([c.weights_flatten().to(self.server.device) for c in self.server.clients])
        D = self.server.architecture.distance_measure(all_clients_weights, shareable_clients_weights)
        for i, receiver in enumerate(self.server.clients):
            d = D[i]
            for j, distance in enumerate(d):
                if i == j:
                    continue
                giver = self.server.clients[j]
                if self._validate_receiver_and_giver(receiver, giver):
                    s = self._calculate_s(distance)
                    self.share_dict[receiver][giver] = np.clip(s, 0, giver.privacy).item()
                    self.share_records[receiver.id][giver.id] = None

    def update(self):
        def _remove_duplicate_data(receiver_a, receiver_b):
            giver_ids_A, giver_ids_B = self.share_records[receiver_a.id].keys(), self.share_records[
                receiver_b.id].keys()
            intersection = giver_ids_A & giver_ids_B
            new_record_a = {}
            if bool(intersection):
                result = []
                index = 0
                for giver_id in giver_ids_A:
                    n_samples = self.share_records[receiver_a.id][giver_id]
                    if giver_id not in intersection:
                        result.append([index, index + n_samples])
                        new_record_a[giver_id] = n_samples
                    index += n_samples
                return result, 1, new_record_a, self.share_records[receiver_b.id].copy()
            return 1, 1, self.share_records[receiver_a.id].copy(), self.share_records[receiver_b.id].copy()

        new_share_dict = {c: {} for c in self.server.clients}
        shape_log = self.server.shape.log
        for node in shape_log.keys():
            client = self.server.node_to_client(node)
            if len(shape_log[node]) == 2:
                id_a, id_b = shape_log[node]
                client_a, client_b = self._id_to_key(id_a), self._id_to_key(id_b)
                s_a, s_b, record_a, record_b = _remove_duplicate_data(client_a, client_b)
                new_share_dict[client] = {client_a: s_a, client_b: s_b}
                self.share_records[client.id] = record_a | record_b
            elif len(shape_log[node]) == 1:
                client_prev = self._id_to_key(shape_log[node][0])
                new_share_dict[client] = {client_prev: 1}
                self.share_records[client.id] = self.share_records[client_prev.id].copy()
        for receiver in new_share_dict:
            receiver_node = self.server.client_to_node(receiver)
            try:
                if receiver_node.is_leaf():
                    continue
            except AttributeError:
                print(receiver)
                print(new_share_dict)
                print(self.server.clients_dict)
                print(self.server.architecture.nodes_dict)
                raise AttributeError
            leaf_clients = [self.server.node_to_client(n) for n in receiver_node.get_leaves()]
            receiver_weight = torch.stack([receiver.weights_flatten().to(self.server.device)])
            for client in leaf_clients:
                if self._validate_receiver_and_giver(receiver, client):
                    client_weight = torch.stack([client.weights_flatten().to(self.server.device)])
                    distance = self.server.architecture.distance_measure(receiver_weight, client_weight)
                    s = self._calculate_s(distance)
                    new_share_dict[receiver][client] = np.clip(s, 0, client.privacy).item()
                    assert new_share_dict[receiver][client] != 0, f"{s}, {client.privacy}"
                    self.share_records[receiver.id][client.id] = None

        self.share_dict = new_share_dict

    def share(self):
        self.server.logger.info("Assigning shared data...")
        data_dict = {c: {} for c in self.server.clients}
        for receiver in self.share_dict.keys():
            for giver, s in self.share_dict[receiver].items():
                n_samples = giver.samples(shareable=True)
                if isinstance(s, (int, float)):
                    data_dict[receiver][giver] = np.random.choice(n_samples, size=int(n_samples * s / giver.privacy),
                                                                  replace=False) if s != 1 else None
                elif isinstance(s, list):
                    if len(s) == 0:
                        pass
                    else:
                        data_dict[receiver][giver] = [i for start, end in s for i in range(start, end)]

        for receiver in data_dict.keys():
            if not receiver.keep_data:
                receiver.train_loader = receiver.backup_train_loader
            subsets = []
            for giver, indices in data_dict[receiver].items():
                subset = Subset(giver.get_shareable_data().dataset, indices) if indices is not None else giver.get_shareable_data().dataset
                subsets.append(subset)
                if giver.id in self.share_records[receiver.id]:
                    self.share_records[receiver.id][giver.id] = len(subset)
            if len(subsets) == 0:
                continue

            assert None not in self.share_records[receiver.id].values(), f"\n{receiver}\n{self.share_records[receiver.id]}\n{self.share_dict[receiver]}\n{data_dict[receiver]}" # DEBUG
            subsets = ([receiver.train_loader.dataset] if receiver.samples() != 0 else []) + subsets
            # datasets_samples = [len(dataset) for dataset in subsets]
            dataset = ConcatDataset(subsets)
            assert len(dataset) == sum(item for item in self.share_records[receiver.id].values()), f"{receiver}\n{len(dataset)}\n{sum(item for item in self.share_records[receiver.id].values())}\n{self.share_records_to_str()}" # DEBUG
            data_loader = DataLoader(
                dataset,
                batch_size=self.server.batch_size,
                shuffle=True
            )
            receiver.train_loader = data_loader
            if not self.server.client_to_node(receiver).is_leaf():
                attrs = []
                max_iters = []
                for client in data_dict[receiver].keys():
                    attrs.append([client.lr, client.weight_decay, client.local_updates])
                    max_iters.append(client.max_iters)
                # TODO: weighted average
                attrs = np.mean(np.array(attrs), axis=0)
                receiver.lr = attrs[0].item()
                receiver.weight_decay = attrs[1].item()
                receiver.local_updates = int(attrs[2].item())
                receiver.max_iters = None if all(item is None for item in max_iters) else int(
                    np.mean([item for item in max_iters if item is not None]).item())
                receiver.privacy = 1
                if receiver.optimizer is None:
                    receiver.optimizer_func = self.server.optimizer
                    receiver.assign_optimizer()

    def backup_train_loader(self):
        for client in self.server.clients:
            if not client.keep_data:
                if hasattr(client, 'backup_train_loader'):
                    continue
                else:
                    client.backup_train_loader = client.train_loader
