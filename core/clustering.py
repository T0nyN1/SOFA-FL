import copy
import io
import itertools
import os.path
from typing import List, Union

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils.utility import euclidean_distance, manhattan_distance, union_find_groups


class Cluster_Node:
    def __init__(self, index, successors=[], level=0, centroid=None, samples=0):
        self.index = index
        self.successors = successors
        self.level = level
        self.centroid = centroid
        self.samples = samples

    def is_leaf(self):
        return len(self.successors) == 0

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for successor in self.successors:
            leaves.extend(successor.get_leaves())
        return leaves

    def get_children(self):
        queue = self.successors.copy()
        children = []
        while queue:
            node = queue.pop(0)
            children.append(node)
            queue.extend(node.successors)
        return children

    def __repr__(self):
        return f"<Cluster_Node(index={self.index}, level={self.level}, centroid={str(self.centroid).replace("\n      ", "")}, samples={self.samples}, successors={[n.index for n in self.successors] if self.successors else None})>"

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return self.index == other.index


class Hierarchical_Clustering:
    def __init__(self, weights_list, samples, type='dmac', distance='euclidean', threshold=0.03, increment_factor=1.1,
                 device='cpu', exp_dir=None, logger=None):
        self.type = type
        self.threshold = threshold
        self.device = device
        self.nodes = [Cluster_Node(i, level=0, centroid=w.to(device), samples=int(sample)) for
                      i, (w, sample) in enumerate(zip(weights_list, samples))]
        self.increment_factor = increment_factor
        self.max_index = len(self.nodes) - 1
        self.exp_dir = exp_dir
        self.nodes_dict = {}
        self.update_nodes_dict()
        self.logger = logger

        if distance == 'euclidean':
            self.distance_measure = euclidean_distance
        elif distance == 'manhattan':
            self.distance_measure = manhattan_distance
        else:
            raise NotImplementedError("Unknown distance measure")

    def __repr__(self):
        s = f'<Hierarchical_Clustering(type={self.type}, distance={self.distance_measure.__name__}, threshold={self.threshold}, increment_factor={self.increment_factor})>'
        s += "\nNodes\n" + "\n".join([node.__repr__() for node in sorted(self.nodes, key=lambda n: n.index)])
        return s

    def update_nodes_dict(self):
        self.nodes_dict = self.clients_dict = {n.index: i for i, n in enumerate(self.nodes)}

    def root(self) -> Cluster_Node:
        return self.find_nodes_of_level(max([n.level for n in self.nodes]))[0]

    def clients(self) -> List[Cluster_Node]:
        return self.find_nodes_of_level(0)

    def add_node(self, node):
        index = self._increment_index()
        node.index = index
        self.nodes.append(node)

    def drop_node(self, node):
        self.nodes.remove(node)

    def fit(self):
        self.get_clustering(self.type)
        self.clustering()
        self.update_nodes_dict()

    def clustering(self):
        clustering = self.get_clustering(self.type)
        clustering()

    def get_clustering(self, type):
        clustering = {'hc': self.agglomerative_clustering, 'dmac': self.dynamic_multi_branch_agglomerative_clustering}
        if type not in clustering.keys():
            raise NotImplementedError("Unknown hierarchical clustering type")
        return clustering[type]

    def _increment_index(self) -> int:
        self.max_index += 1
        return self.max_index

    def find_centroid(self, nodes, return_n_samples=True) -> Union[torch.Tensor, int]:
        if nodes is None or len(nodes) == 0:
            centroid, samples = 0, 0
        else:
            centroids_all = torch.stack([n.centroid for n in nodes], dim=0)
            samples_all = torch.tensor([[n.samples] for n in nodes], dtype=torch.float32, device=self.device)
            weighted_sum = torch.sum(centroids_all * samples_all, dim=0)
            total_samples = torch.sum(samples_all)
            centroid = weighted_sum / total_samples
            samples = int(total_samples.item())
        if return_n_samples:
            return centroid, samples
        else:
            return centroid

    def _matrix_min_max_normalization(self, matrix, mask=None):
        valid_vals = matrix if mask is None else matrix[mask]
        min_val = valid_vals.min()
        max_val = valid_vals.max()

        if max_val > min_val:
            matrix_norm = (matrix - min_val) / (max_val - min_val)
        else:
            matrix_norm = matrix.clone()
        return matrix_norm

    def find_nodes_of_level(self, level: int) -> List[Cluster_Node]:
        return [n for n in self.nodes if n.level == level]

    def find_predecessor(self, node):
        if node == self.root():
            return None
        for n in self.nodes:
            if not n.is_leaf() and node in n.successors:
                return n

    def normalize_levels(self):
        s = []
        levels = sorted(list({n.level for n in self.nodes}))
        current_level = -1
        for level in levels:
            current_level += 1
            if level == current_level:
                continue
            else:
                s.append(f"{level} -> {current_level}")
                nodes = self.find_nodes_of_level(level)
                for node in nodes:
                    node.level = current_level
        return "Normalize levels: " + ", ".join(s) if len(s) > 0 else ""

    def update_attrs(self):
        for node in sorted(self.nodes, key=lambda n: n.level):
            if node.is_leaf():
                continue
            node.centroid, node.samples = self.find_centroid(node.successors)

    def dynamic_multi_branch_agglomerative_clustering(self):
        nodes = copy.deepcopy(self.nodes)
        level = 0

        initial_count = len(nodes)
        with tqdm(total=1.0, desc="Clustering Progress", ncols=80, bar_format='{l_bar}{bar}| {n:.1%}') as pbar:
            threshold = self.threshold
            while len(nodes) > 1:
                remaining = [n.index for n in nodes]
                level += 1
                centroids = torch.stack([n.centroid for n in nodes])
                D = self.distance_measure(centroids)
                mask = ~torch.eye(D.size(0), dtype=torch.bool, device=self.device)
                D = self._matrix_min_max_normalization(D, mask)

                mask = torch.triu(D < threshold, diagonal=1)
                pairs = [[remaining[i.item()], remaining[j.item()]] for i, j in torch.nonzero(mask, as_tuple=False)]
                if len(pairs) == 0:
                    threshold *= self.increment_factor
                    level -= 1
                    continue
                else:
                    threshold = self.threshold

                merge_groups = union_find_groups(pairs)
                remaining = list(set(remaining) - set(itertools.chain.from_iterable(merge_groups)))
                nodes = [self.nodes[i] for i in remaining]

                for group in merge_groups:
                    index = self._increment_index()
                    group_nodes = [self.nodes[i] for i in group]
                    centroid, total_samples = self.find_centroid(group_nodes)
                    cluster = Cluster_Node(index=index, successors=group_nodes, level=level, centroid=centroid,
                                           samples=total_samples)
                    self.nodes.append(cluster)
                    nodes.append(cluster)
                    remaining.append(index)
            progress = 1 - (len(nodes) - 1) / initial_count
            pbar.n = progress
            pbar.refresh()

    def agglomerative_clustering(self, method: str = 'average'):
        """
        Standard binary agglomerative clustering using SciPy linkage.
        Always merges two closest clusters at a time.

        Args:
            method: linkage method ('single', 'complete', 'average', 'ward', etc.)
        """
        # TODO: formating
        centroids = torch.stack([n.centroid for n in self.nodes]).cpu().numpy()

        if self.distance_measure.__name__ == 'euclidean_distance':
            metric = 'euclidean'
        elif self.distance_measure.__name__ == 'manhattan_distance':
            metric = 'cityblock'
        Z = linkage(centroids, method=method, metric=metric)

        n = len(self.nodes)
        current_nodes = list(self.nodes)
        level = 0

        for k, (a, b, dist, cnt) in enumerate(Z):
            level += 1

            def get_node(i):
                return current_nodes[int(i)] if i < len(current_nodes) else new_nodes[int(i) - len(current_nodes)]

            if k == 0:
                new_nodes = []

            node_a = get_node(a)
            node_b = get_node(b)
            centroid, total_samples = self.find_centroid([node_a, node_b])
            index = self._increment_index()

            cluster = Cluster_Node(
                index=index,
                successors=[node_a, node_b],
                level=level,
                centroid=centroid,
                samples=total_samples
            )

            self.nodes.append(cluster)
            new_nodes.append(cluster)

    def visualize_tree(self, **kwargs):
        positions = {}
        leaf_x_counter = 0

        root = self.root()
        max_level = root.level

        def assign_positions(node):
            """
            Assigns (x, y) coordinates to a node and its descendants.
            Returns (min_leaf_x, max_leaf_x) of the subtree rooted at this node.
            """
            nonlocal leaf_x_counter
            y = node.level
            if node.is_leaf():
                x = leaf_x_counter
                leaf_x_counter += 1
                positions[node.index] = (x, y)
                return x, x
            else:
                child_spans = [assign_positions(child) for child in node.successors]
                min_leaf_x = min(span[0] for span in child_spans)
                max_leaf_x = max(span[1] for span in child_spans)
                x = (min_leaf_x + max_leaf_x) / 2.0

                positions[node.index] = (x, y)
                return min_leaf_x, max_leaf_x

        assign_positions(root)
        edge_x = []
        edge_y = []
        node_x = []
        node_y = []
        node_labels = []
        node_hover_text = []

        node_map = {n.index: n for n in self.nodes}

        for node_index, (x, y) in positions.items():
            node = node_map[node_index]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(f"{node.index}")
            node_hover_text.append(
                f"ID: {node.index}<br>Level: {node.level}<br>Samples: {node.samples}<br>Successors: {[n.index for n in node.successors] if not node.is_leaf() else None}"
            )

        for node in self.nodes:
            if not node.is_leaf():
                px, py = positions[node.index]
                for child in node.successors:
                    cx, cy = positions[child.index]

                    edge_x.extend([px, cx, None])
                    edge_y.extend([py, cy, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="gray"),
            hoverinfo="none",
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            hovertext=node_hover_text,
            hoverinfo="text",
            marker=dict(size=10, color="cornflowerblue", line=dict(width=1, color="black")),
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title="Hierarchical Clustering Tree (Dynamic Multi-Branch)",
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(
                title="Hierarchy Level (inverted)",
                showgrid=False,
                zeroline=False,
                range=[-0.5, max_level + 0.75]
            ),
            plot_bgcolor="white",
            hovermode="closest",
            height=600,
            margin=dict(t=100, b=50, l=50, r=50)
        )
        name = kwargs.get("name", "dendrogram")
        save_dir = kwargs.get("save_dir", self.exp_dir)

        if kwargs.get("save_plot", False):
            extension = kwargs.get("format", "html")
            if extension == "html":
                fig.write_html(os.path.join(save_dir, f"{name}.html"))
            else:
                try:
                    fig.write_image(os.path.join(save_dir, f"{name}.{extension}"), width=kwargs.get("width", 1000),
                                    height=kwargs.get("height", 600))
                except:
                    self.logger.warning(f"Could not write image in format: {extension}, changed to html")
                    fig.write_html(os.path.join(save_dir, f"{name}.html"))

        if kwargs.get("show", False):
            fig.show()

        if kwargs.get("return_img", False):
            img_bytes = fig.to_image(format="png", width=kwargs.get("width", 1000), height=kwargs.get("height", 600),
                                     scale=kwargs.get("scale", 1))

            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            return img_array


class SHAPE:
    '''implements the Self-organizing Hierarchical Adaptive Propagation and Evolution (SHAPE) algorithm'''

    def __init__(self, tree: Hierarchical_Clustering, graft_tolerance=0.1, split_threshold=0.5, merge_threshold=0.5,
                 max_splits=3):
        self.tree: Hierarchical_Clustering = tree
        self.split_threshold = split_threshold
        self.graft_tolerance = graft_tolerance
        self.merge_threshold = merge_threshold
        self.max_splits = max_splits
        self.log: dict = None

    def find_lowest_common_ancestor(self, root, node_a, node_b):
        if root is None:
            return None

        if root == node_a or root == node_b:
            return root
        matches = []
        for child in root.successors:
            res = self.find_lowest_common_ancestor(child, node_a, node_b)
            if res is not None:
                matches.append(res)

        if len(matches) >= 2:
            return root

        return matches[0] if matches else None

    def incoherence(self, node: Cluster_Node):
        # TODO: whether to trim cluster with no successors? Why no successors happened here?
        successors = node.successors
        if successors is None or len(successors) == 0:
            return 0
        successors_weights = torch.stack([n.centroid for n in node.successors]).to(self.tree.device)
        D = self.tree.distance_measure(successors_weights, node.centroid.view(1, -1))
        D = self.tree._matrix_min_max_normalization(D)
        return torch.mean(D).item()

    def add(self, node, predecessor, log_value):
        if predecessor is not None:
            predecessor.successors.append(node)
        self.tree.add_node(node)
        self.log[node] = log_value

    def drop(self, node):
        predecessor = self.tree.find_predecessor(node)
        if predecessor is not None:
            predecessor.successors.remove(node)
        self.tree.drop_node(node)
        if node in self.log.keys() and len(self.log[node]) == 2:
            for key, value in self.log.items():
                if len(value) == 1 and value[0] == node.index:
                    self.log[key] = self.log[node]
        self.log[node] = []

    def merge(self, group: List[Cluster_Node]):
        assert all(not node.is_leaf() for node in group)
        assert all(
            self.tree.find_predecessor(node) == (predecessor := self.tree.find_predecessor(group[0])) for node in group)
        if not all(node.level == group[0].level for node in group):
            return
        new_node = Cluster_Node(-1, [s for node in group for s in node.successors], group[0].level, None, None)
        new_node.centroid, new_node.samples = self.tree.find_centroid(new_node.successors)
        for node in group:
            self.drop(node)
        self.add(new_node, predecessor, [n.index for n in group])
        return new_node

    def split(self, node: Cluster_Node, k_max=3):
        assert not node.is_leaf()
        data_points = torch.stack([s.centroid.detach() for s in node.successors])
        data_numpy = data_points.cpu().numpy()
        original_successors = list(node.successors)
        for k in range(2, k_max + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=None)
            labels = kmeans.fit_predict(data_numpy)
            new_groups = [[] for _ in range(k)]

            for i, successor_node in enumerate(original_successors):
                new_groups[labels[i]].append(successor_node)
            new_nodes = [Cluster_Node(None, group, node.level, None, None) for group in new_groups if len(group) > 0]
            for n in new_nodes:
                n.centroid, n.samples = self.tree.find_centroid(n.successors)
            satisfied = all(self.incoherence(n) < self.split_threshold for n in new_nodes) or k == k_max
            if not satisfied:
                continue

            predecessor = self.tree.find_predecessor(node)
            for n in new_nodes:
                self.add(n, predecessor, [node.index])

            self.drop(node)
            return new_nodes

    def _recur_update_attrs(self, lca, node_a, node_b):
        def update_upwards(n, stop):
            while n is not None and n != stop:
                n.centroid, n.samples = self.tree.find_centroid(n.successors)
                n = self.tree.find_predecessor(n)

        update_upwards(node_a, lca)
        update_upwards(node_b, lca)

    def graft(self, child: Cluster_Node, parent: Cluster_Node):
        assert not parent.is_leaf()
        predecessor = self.tree.find_predecessor(child)
        ancestor = self.find_lowest_common_ancestor(self.tree.root(), parent, predecessor)
        predecessor.successors.remove(child)
        parent.successors.append(child)
        # self._recur_update_attrs(ancestor, predecessor, parent)

    def trim(self, node: Cluster_Node):
        assert len(node.successors) == 1
        successor = node.successors[0]
        predecessor = self.tree.find_predecessor(node)
        predecessor.successors.append(successor)
        self.drop(node)

    def collapse(self):
        root = self.tree.root()
        assert len(root.successors) == 1
        node = root.successors[0]
        new_root = Cluster_Node(None, node.successors.copy(), root.level, root.centroid, root.samples)
        self.add(new_root, None, [root.index, node.index])
        self.drop(root)
        self.drop(node)

    def run(self):
        self.log = {}
        max_level = self.tree.root().level
        for l in range(max_level - 1):
            nodes = self.tree.find_nodes_of_level(l)
            if len(nodes) == 0:
                continue
            nodes_centroids = torch.stack([n.centroid for n in nodes]).to(self.tree.device)
            parents = self.tree.find_nodes_of_level(l + 1)
            # TODO: level completed deleted.
            if len(parents) == 0:
                continue
            parents_centroids = torch.stack([n.centroid for n in parents]).to(self.tree.device)
            D = self.tree.distance_measure(nodes_centroids, parents_centroids)
            for i, node in enumerate(nodes):
                d_pre = self.tree.distance_measure(node.centroid.view(1, -1),
                                                   (ex_parent := self.tree.find_predecessor(node)).centroid.view(1,
                                                                                                                 -1)).item()
                d_min = torch.min(D[i]).item()
                if d_min * (1 + self.graft_tolerance) > d_pre:
                    continue
                parent = parents[torch.argmin(D[i])]
                self.tree.logger.info(f"Graft: {node.index}: {ex_parent.index} -> {parent.index}")
                self.graft(node, parent)

        for l in range(1, max_level - 1):
            nodes = self.tree.find_nodes_of_level(l)
            if len(nodes) == 0:
                continue
            centroids = torch.stack([n.centroid for n in nodes]).to(self.tree.device)
            D = self.tree.distance_measure(centroids)
            mask = ~torch.eye(D.size(0), dtype=torch.bool, device=self.tree.device)
            D = self.tree._matrix_min_max_normalization(D, mask)
            mask = torch.triu(D < self.merge_threshold, diagonal=1)
            pairs = [[nodes[i.item()], nodes[j.item()]] for i, j in torch.nonzero(mask, as_tuple=False)]
            pairs = [[a, b] for a, b in pairs if self.tree.find_predecessor(a) == self.tree.find_predecessor(b)]
            groups = union_find_groups(pairs)
            for group in groups:
                self.tree.logger.info(f"Merge: {" + ".join([str(n.index) for n in group])} -> {node.index}")

        for l in range(1, max_level):
            nodes = self.tree.find_nodes_of_level(l)
            for node in nodes:
                if self.incoherence(node) > self.split_threshold and len(
                        node.successors) >= 2:
                    new_nodes = self.split(node, self.max_splits)
                    self.tree.logger.info(f"Split: {node.index} -> {[n.index for n in new_nodes]}")

        for node in self.tree.nodes.copy():
            if node.is_leaf() or node == self.tree.root():
                continue
            if len(node.successors) == 1:
                self.tree.logger.info(f"Trim: {node.index}")
                self.trim(node)

        if len(self.tree.root().successors) == 1:
            self.collapse()
            self.tree.logger.info(f"Collapse: {self.tree.root().index}")

        record = self.tree.normalize_levels()
        if record is not None:
            self.tree.logger.info(record)

        self.tree.update_nodes_dict()
