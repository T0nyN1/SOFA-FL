import os.path
from itertools import zip_longest

import imageio.v2 as imageio
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

from core.loss import SOFA_FL_Loss
from utils.utility import is_required

title_dict = {"train_losses": "Training Loss",
              "local_losses": "Local Objective",
              "inter_cluster_losses": "Inter Cluster Loss",
              "intra_cluster_losses": "Intra Cluster Loss"}


# TODO: train progress log, currently unimplemented.

class Base_Manager():
    def __init__(self, server, logger):
        self.server = server
        self.parallel = self.server.cfg['experiment']['parallel_training']
        self.logger = logger

        self.round = 0

    def get_local_objective(self, key):
        local_objective_dict = {'cross_entropy': F.cross_entropy,
                                'mse': F.mse_loss,
                                'mae': F.l1_loss,
                                'ce': F.cross_entropy, }
        if key not in local_objective_dict.keys():
            raise ValueError('Unknown local objective {}'.format(key))
        return local_objective_dict[key]

    def _instantiate_criterion(self, client, device=None):
        local_objective = self.get_local_objective(self.server.cfg['train']['local_objective'])
        criterion = SOFA_FL_Loss(local_objective, self.server.cfg['train']['alpha'],
                                 self.server.cfg['train']['beta'])
        predecessor_node = self.server.structure.find_predecessor(self.server.client_to_node(client))
        if predecessor_node is None:
            predecessor_weight = None
            siblings_weights = None
        else:
            predecessor_client = self.server.node_to_client(predecessor_node.index)
            predecessor_weight = predecessor_client.weights_flatten()
            siblings_weights = [self.server.node_to_client(n).weights_flatten() for n in predecessor_node.successors]
            if device is not None:
                predecessor_weight = predecessor_weight.to(device)
                siblings_weights = [w.to(device) for w in siblings_weights]
        return criterion, siblings_weights, predecessor_weight

    def train_one_round(self):
        losses = {"train_losses": {}, "local_losses": {}, "inter_cluster_losses": {}, "intra_cluster_losses": {}}
        for client in tqdm(self.server.clients, desc=f"Round {self.round} | Clients local updating: "):
            if client.samples() == 0:
                continue

            id, updates, loss = client.local_update(*self._instantiate_criterion(client))
            client.updates = updates
            self.server.sync_client_weights(client)
            for k, v in loss.items():
                losses[k][id] = v
        return losses

    # TODO: cuda support
    @staticmethod
    def _client_update_task(task):
        client, device_str, criterion, siblings_clients, predecessor_client = task
        torch.set_num_threads(1)
        device = torch.device(device_str)
        client.model.to(device)
        client.assign_optimizer()
        siblings_clients = [w.to(device) for w in siblings_clients] if siblings_clients is not None else None
        predecessor_client = predecessor_client.to(device) if predecessor_client is not None else None
        client_id, updates, losses = client.local_update(criterion, siblings_clients, predecessor_client)

        updated_weights = {k: v.cpu() for k, v in client.model.state_dict().items()}
        updates = {k: v.cpu() for k, v in updates.items()}

        return client_id, losses, updated_weights, updates

    def train_one_round_parallel(self):
        MAX_WORKERS = self.server.cfg['experiment']['num_workers']
        selected_clients = [client for client in self.server.clients if client.samples() > 0]
        tasks = []

        for client in selected_clients:
            client.model.to('cpu')
            client.updates.to('cpu')
            client.model.zero_grad(set_to_none=True)
            client.optimizer = None

            if isinstance(client.w, torch.Tensor):
                client.w = client.w.cpu()

            tasks.append((client, str(self.server.device), *self._instantiate_criterion(client, device='cpu')))

        losses = {"train_losses": {}, "local_losses": {}, "inter_cluster_losses": {}, "intra_cluster_losses": {}}

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        with mp.Pool(processes=MAX_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(self._client_update_task, tasks),
                total=len(tasks),
                desc=f"Round {self.round} | Parallel Updating"
            ))

        for client_id, loss, new_weights, updates in results:
            for k, v in loss.items():
                losses[k][client_id] = v
            target_client = next((c for c in self.server.clients if c.id == client_id), None)
            if target_client:
                target_client.model.load_state_dict(new_weights)
                target_client.updates.load_state_dict(updates)
                target_client.model.to(self.server.device)  # sent back to gpu, if needed
                self.server.sync_client_weights(target_client)

        return losses

    def _next_round(self):
        if self.parallel:
            losses = self.train_one_round_parallel()
        else:
            losses = self.train_one_round()
        return losses

    def _init_clustering(self):
        self.server.structure.fit()
        self.server.initialize_clients()
        if self.server.data_share:
            self.server.data_share_manager.backup_train_loader()
            self.server.data_share_manager.initialize()
            self.server.data_share_manager.share()

    def _update_clustering(self):
        self.server.shape.run()
        self.server.update_clients()
        if self.server.data_share:
            self.server.data_share_manager.update()
            self.server.data_share_manager.share()

    def evaluate(self):
        accuracies = {}
        if not self.parallel:
            for client in tqdm(self.server.clients, desc=f"Evaluation: "):
                id, accuracy = client.evaluate()
                accuracies[id] = accuracy
        else:
            accuracies = self.evaluate_parallel()

        return accuracies

    @staticmethod
    def _eval_task(task):
        client, device_str = task
        torch.set_num_threads(1)
        device = torch.device(device_str)

        client.model.to(device)
        client.device = device

        with torch.no_grad():
            client_id, accuracy = client.evaluate()
        client.model.to('cpu')

        return client_id, accuracy

    def evaluate_parallel(self):
        MAX_WORKERS = self.server.cfg['experiment']['num_workers']

        target_device = 'cpu'

        tasks = []

        for client in self.server.clients:
            client.model.to('cpu')
            client.updates.to('cpu')
            client.optimizer = None

            if hasattr(client, 'w') and isinstance(client.w, torch.Tensor):
                client.w = client.w.cpu()

            tasks.append((client, target_device))

        accuracies = {}

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        with mp.Pool(processes=MAX_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(self._eval_task, tasks),
                total=len(tasks),
                desc=f"Evaluation (Parallel)"
            ))

        for client_id, accuracy in results:
            accuracies[client_id] = accuracy

        return accuracies

    # Overwrite in subclasses.
    def step(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class Train_Manager(Base_Manager):
    def __init__(self, server, logger, val=False):
        super().__init__(server, logger)
        self.val = val

        self.export_type = self.server.cfg['output']['export_statistics']

        self.show_plots = self.server.cfg['output']['show_plots']
        self.show_structure = is_required("structure", self.show_plots)

        self.save_plots = self.server.cfg['output']['save_plots']
        self.format = self.server.cfg['output']['format']
        self.w, self.h = self.server.cfg['output']['image_size']
        self.duration = self.server.cfg['output']['duration']
        self.save_structure = is_required("structure", self.save_plots)

        self.dendrograms = []
        self.stats = {"train_losses": [], "val_accuracies": [], "local_losses": [], "inter_cluster_losses": [],
                      "intra_cluster_losses": []}

    def _update_stats(self, key, values):
        self.stats[key].append(values)

    def _warm_up(self):
        # Warm-up before first clustering
        # FIXME: Bug exists. Fail to work.
        warmup_iters = self.server.cfg['server']['scheme']['rounds_before_first_clustering']
        if warmup_iters is None:
            warmup_iters = 1
            for _ in range(warmup_iters):
                losses = self._next_round()
                for k, v in losses.items():
                    self._update_stats(k, v)

    def step(self):
        self.round += 1
        losses = self._next_round()

        for k, v in losses.items():
            self._update_stats(k, v)
        self._update_clustering()
        if self.save_structure:
            img = self.server.structure.visualize_tree(return_img=self.save_structure,
                                                       width=self.w,
                                                       height=self.h,
                                                       show=self.show_structure)
            self.dendrograms.append(img)
        elif self.show_structure:
            self.server.structure.visualize_tree(show=True)

        if self.val:
            accuracies = self.evaluate()
            self._update_stats("val_accuracies", accuracies)

    def run(self):
        def _console_output(gap=10):
            server_str = self.server.__repr__().split('\n')
            arch_str = self.server.structure.__repr__().split('\n')
            length = max([len(line) for line in server_str])
            info = [f"{a:<{length + gap}}{b}" for a, b in zip_longest(server_str, arch_str, fillvalue="")]
            self.logger.info(f"Round {self.round}:\n{"\n".join(info)}")
            if self.server.data_share:
                self.logger.info(self.server.data_share_manager)

        self.logger.info("Start training..." if not self.parallel else "Start training in parallel...")
        self.logger.info("Warming up...")
        self._warm_up()
        if self.val:
            accuracies = self.evaluate()
            self._update_stats("val_accuracies", accuracies)
        self.logger.info("Forming server structure...")
        self._init_clustering()
        if self.save_structure:
            img = self.server.structure.visualize_tree(return_img=self.save_structure,
                                                       width=self.w,
                                                       height=self.h,
                                                       show=self.show_structure)
            self.dendrograms.append(img)
        elif self.show_structure:
            self.server.structure.visualize_tree(show=True)

        _console_output()
        for i in range(self.server.comm_rounds - 1):
            self.step()
            _console_output()
        self.visualize()
        self.export()
        return self.stats

    def visualize_train_losses(self, loss_type, **kwargs):
        train_losses_data = self.stats.get(loss_type, None)

        if train_losses_data is None or not train_losses_data:
            print("No training loss data to visualize.")
            return

        fig = go.Figure()
        num_rounds = len(train_losses_data)

        all_client_ids = set()
        for r_data in train_losses_data:
            all_client_ids.update(r_data.keys())

        try:
            sorted_client_ids = sorted(list(all_client_ids), key=lambda x: int(x))
        except:
            sorted_client_ids = sorted(list(all_client_ids), key=str)

        for client_id in sorted_client_ids:
            client_x_coords = []
            client_y_losses = []

            for r_idx in range(num_rounds):
                round_data = train_losses_data[r_idx]

                if client_id in round_data:
                    losses = round_data[client_id]
                    num_updates = len(losses)

                    if num_updates > 0:
                        x_coords = [r_idx + (k + 0.5) / num_updates for k in range(num_updates)]

                        client_x_coords.extend(x_coords)
                        client_y_losses.extend(losses)
                else:
                    if client_x_coords and client_x_coords[-1] is not None:
                        client_x_coords.append(None)
                        client_y_losses.append(None)

            fig.add_trace(go.Scatter(
                x=client_x_coords,
                y=client_y_losses,
                mode='lines+markers',
                name=f'Client {client_id}',
                marker=dict(size=4),
                line=dict(width=1.5),

                hovertemplate=f'<b>Client {client_id}</b><br>' +
                              'Global Round: %{x:.2f}<br>' +
                              'Loss: %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title={
                'text': f"{title_dict[loss_type]}",
                'y': 0.95, 'x': 0.5,
                'xanchor': 'center', 'yanchor': 'top',
                'font': dict(size=18, color='black')
            },
            xaxis_title={
                'text': "Communication Rounds",
                'font': dict(size=14, color='black')
            },
            yaxis_title={
                'text': "Training Loss",
                'font': dict(size=14, color='black')
            },
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                title="Clients",
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="Black",
                borderwidth=0
            ),
            margin=dict(t=80, b=80, l=80, r=80)
        )

        axis_tickvals = [r + 0.5 for r in range(num_rounds)]
        axis_ticktext = [f'{r}' for r in range(num_rounds)]

        if num_rounds > 30:
            step = num_rounds // 15
            axis_tickvals = axis_tickvals[::step]
            axis_ticktext = axis_ticktext[::step]

        fig.update_xaxes(
            tickvals=axis_tickvals,
            ticktext=axis_ticktext,
            showgrid=False
        )

        for r in range(num_rounds + 1):
            fig.add_shape(
                type="line",
                x0=r, x1=r,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash")
            )

        fig.update_yaxes(showgrid=True, gridcolor="rgba(220,220,220,0.7)")

        name = kwargs.get("name", loss_type)
        save_dir = kwargs.get("save_dir", self.server.cfg['experiment']['experiment_dir'])

        if kwargs.get("save_plot", False):
            extension = kwargs.get("format", "html")
            if extension == "html":
                fig.write_html(os.path.join(save_dir, f"{name}.html"))
            else:
                try:
                    fig.write_image(os.path.join(save_dir, f"{name}.{extension}"), width=kwargs.get("width", 800),
                                    height=kwargs.get("height", 600))
                except:
                    self.logger.warning(f"Could not write image in format: {extension}, changed to html")
                    fig.write_html(os.path.join(save_dir, f"{name}.html"))

        if kwargs.get("show", False):
            fig.show()

    def visualize_val_accuracy(self, **kwargs):
        val_data_by_round = self.stats["val_accuracies"]

        if not val_data_by_round:
            print("No validation data to visualize.")
            return

        fig = go.Figure()

        num_rounds = len(val_data_by_round)
        x_axis_rounds = list(range(num_rounds))

        all_client_ids = set()
        for round_data in val_data_by_round:
            all_client_ids.update(round_data.keys())

        try:
            sorted_client_ids = sorted(list(all_client_ids), key=lambda x: int(x))
        except:
            sorted_client_ids = sorted(list(all_client_ids), key=str)

        for client_id in sorted_client_ids:
            client_accs = []
            for r_idx in range(num_rounds):
                acc = val_data_by_round[r_idx].get(client_id, None)
                client_accs.append(acc)

            fig.add_trace(go.Scatter(
                x=x_axis_rounds,
                y=client_accs,
                mode='lines+markers',
                name=f'Client {client_id}',
                # connectgaps=True, # Cancel if breaks between points are unwanted.
                marker=dict(size=6, opacity=0.8),
                line=dict(width=2),
                hovertemplate=f'<b>Client {client_id}</b><br>' +
                              'Round: %{x}<br>' +
                              'Accuracy: %{y:.3f}' +
                              '<extra></extra>'
            ))

        fig.update_layout(
            title={
                'text': "Federated Validation Accuracy per Client",
                'y': 0.95, 'x': 0.5,
                'xanchor': 'center', 'yanchor': 'top',
                'font': dict(size=18, color='black')
            },
            xaxis_title={
                'text': "Communication Rounds",
                'font': dict(size=14, color='black')
            },
            yaxis_title={
                'text': "Accuracy",
                'font': dict(size=14, color='black')
            },
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                title="Clients",
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="Black",
                borderwidth=0
            ),
            margin=dict(t=80, b=80, l=80, r=80)
        )

        fig.update_xaxes(
            tickvals=x_axis_rounds if num_rounds < 30 else None,
            showgrid=True,
            gridcolor="rgba(220,220,220,0.7)",
            tickfont=dict(size=12, color='black'),
            ticks="outside",
            ticklen=5,
            tickwidth=1,
            title_standoff=15
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(220,220,220,0.7)",
            tickfont=dict(size=12, color='black'),
            ticks="outside",
            ticklen=5,
            tickwidth=1,
            title_standoff=15,
            zeroline=False
        )

        name = kwargs.get("name", "val_accuracies")
        save_dir = kwargs.get("save_dir", self.server.cfg['experiment']['experiment_dir'])

        if kwargs.get("save_plot", False):
            extension = kwargs.get("format", "html")
            if extension == "html":
                fig.write_html(os.path.join(save_dir, f"{name}.html"))
            else:
                try:
                    fig.write_image(os.path.join(save_dir, f"{name}.{extension}"), width=kwargs.get("width", 800),
                                    height=kwargs.get("height", 600))
                except:
                    self.logger.warning(f"Could not write image in format: {extension}, changed to html")
                    fig.write_html(os.path.join(save_dir, f"{name}.html"))

        if kwargs.get("show", False):
            fig.show()

    def visualize_clustering_evolution(self, dendrograms, name="clustering_evolution", duration=0.5):
        imageio.mimsave(os.path.join(self.server.cfg['experiment']['experiment_dir'], f"{name}.gif"), dendrograms,
                        duration=duration)

    def visualize(self):
        if self.save_structure:
            self.server.structure.visualize_tree(save_plot=True,
                                                 format=self.format,
                                                 width=self.w,
                                                 height=self.h)
            self.visualize_clustering_evolution(self.dendrograms, duration=self.server.cfg["output"]["duration"])

        for key in self.stats:
            if key == "val_accuracies" or "model_distribution":
                continue
            if is_required(key, self.save_plots):
                self.visualize_train_losses(key,
                                            show=is_required(key, self.show_plots),
                                            save_plot=True,
                                            format=self.format,
                                            width=self.w,
                                            height=self.h)
            elif is_required(key, self.show_plots):
                self.visualize_train_losses(key, show=True)
        if is_required("model_distribution", self.save_plots):
            self.server.visualize_model_distribution(show=is_required("model_distribution", self.show_plots),
                                                     save_plot=True,
                                                     format=self.format,
                                                     width=self.w,
                                                     height=self.h)
        elif is_required("model_distribution", self.show_plots):
            self.server.visualize_model_distribution(show=True)

        if self.val:
            if is_required("val_accuracies", self.save_plots):
                self.visualize_val_accuracy(show=is_required("val_accuracies", self.show_plots),
                                            save_plot=True,
                                            format=self.format,
                                            width=self.w,
                                            height=self.h)
            elif is_required("val_accuracies", self.show_plots):
                self.visualize_val_accuracy(show=True)

    def export_stats_to_csv(self, stats_type, **kwargs):
        data = self.stats.get(stats_type, None)

        if data is None or not data:
            print("No data to export.")
            return

        results = {"Rounds": []}
        client_ids = sorted(list({client_id for sub_dict in data for client_id in sub_dict.keys()}))
        for client_id in client_ids:
            results[f"Client {client_id}"] = []
        for round, records in enumerate(data):
            max_length = max([len(item) for item in records.values()]) if (
                is_list := stats_type != "val_accuracies") else 1
            results["Rounds"].extend([round] + [None] * (max_length - 1))
            for client_id in client_ids:
                record = records.get(client_id, None)
                if record is None:
                    results[f"Client {client_id}"].extend([None] * max_length)
                    continue
                if not is_list:
                    record = [record]
                results[f"Client {client_id}"].extend(record + [None] * (max_length - len(record)))
        results = pd.DataFrame(results)
        results["Rounds"] = results["Rounds"].astype("Int32")

        name = kwargs.get("name", stats_type)
        save_dir = kwargs.get("save_dir", self.server.cfg['experiment']['experiment_dir'])
        results.to_csv(os.path.join(save_dir, f"{name}.csv"), index=False, encoding="utf-8")

    def export(self):
        for key in self.stats:
            if is_required(key, self.export_type):
                self.export_stats_to_csv(key)


class Evaluation_Manager(Base_Manager):
    def __init__(self, server, logger):
        super().__init__(server, logger)

    def step(self):
        pass

    def run(self):
        self.evaluate()
