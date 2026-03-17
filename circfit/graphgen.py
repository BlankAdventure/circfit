# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:12:17 2026

@author: BlankAdventure

"""

import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
from collections.abc import Hashable, Callable
from pathlib import Path

layouts: dict[str, Callable] = {
        "arf": nx.arf_layout,
        "circ": nx.circular_layout,
        "force": nx.forceatlas2_layout,
        "kamada": nx.kamada_kawai_layout,
        "planar": nx.planar_layout,        
        "shell": nx.shell_layout,
        "spring": nx.spring_layout,
        "spectral": nx.spectral_layout,
        "spiral": nx.spiral_layout,        
        }

nm = isomorphism.categorical_node_match("conn", None)

labels = {0: "i", 1: "o", 2: "g"}

attrs = {
    "i": {"conn": "i"},
    "o": {"conn": "o"},
    "g": {"conn": "g"},
}


class Graphgen():
    def __init__(self, n_edges):
        self.n_edges = n_edges
        self.atlas: list[nx.Graph] = []
        self.circuits: dict[int,list[nx.MultiGraph]] = {}
    
    def build_atlas(self) -> None:
        self.atlas = build_atlas(self.n_edges)

    def build_circuits(self) -> None:
        idx = range(0,len(self.atlas))                
        self.circuits = {i: build_circuits(self.atlas, self.n_edges, indices=[i]) for i in idx}

    def build_all(self) -> None:
        self.build_atlas()
        self.build_circuits()

    def draw_atlas(self, layout=None) -> None:
        if layout:
            multidraw(self.atlas,layout=layout)
        else:
            multidraw(self.atlas)
    
    def save_images(self,path):
        for key, graphs in self.circuits.items():
            for idx, g in enumerate(graphs):
                path = Path(path)
                full_path = path / f"GRAPH-{key}-{idx}.png"
                print(full_path)
                draw(g,save_file=full_path)


def find_bounds(n_edges: int) -> tuple[int, int, int, int]:

    # The nx graph atlas includes all graphs with 1 to 7 nodes. We require a 
    # minium of two, so we start there. 
    nodes = range(2, 8)
    
    # The minimum edges obtain from a node n. 
    min_edges = [n - 1 for n in nodes]    
    max_edges = [n * (n - 1) for n in nodes]    
    bounds = list(zip(nodes, min_edges, max_edges))

    min_edges_p = int(np.ceil(n_edges / 2))

    for idx, lower, upper in bounds:
        if lower <= min_edges_p <= upper:
            break

    print(f"Min Nodes: {idx}")
    print(f"Max Nodes: {n_edges + 1}")
    print(f"Min Edges: {min_edges_p}")
    print(f"Max Edges: {n_edges}")

    return (idx, n_edges + 1, min_edges_p, n_edges)

def to_rlc(G: nx.Graph) -> nx.MultiGraph:
    M: nx.MultiGraph = nx.MultiGraph(G)
    edge_data = list(M.edges(data=True, keys=True))
    for u, v, key, data in edge_data[:]:
        if data["type"] == "P":
            M[u][v][key]["type"] = "L"
            M.add_edge(u, v, type="C")
    return M


def d2_neighbors(G: nx.Graph, node: Hashable) -> tuple[str, str]:
    """
    Returns the two incident edges expected for a degree-2 node.
    """

    edges: list[tuple] = list(G.edges(node, data="type"))
    if len(edges) == 2:
        e1 = edges[0][2]
        e2 = edges[1][2]
    else:
        raise ValueError
    return (e1, e2)


def validate_edge_count(G: nx.Graph, n_expected: int) -> bool:
    """
    Counts the total number of edges in graph G accounting for double
    edges (P-edges), and returns True/False if this matches n_expected count.
    This is used as a filter to remove graphs with an incorrect edge count.
    """

    count = 0
    for _, _, data in G.edges(data=True):
        if data["type"] == "P":
            count += 2
        else:
            count += 1
    return count == n_expected


def simplify(G: nx.Graph) -> nx.Graph:
    """Simplifies graph G by reducing series edges into a single edge"""
    repeat = True
    while repeat:
        extras = internal_nodes(G)
        for node, deg in extras:
            if deg == 2:
                t1, t2 = d2_neighbors(G, node)
                if (t1 == t2 == "L") or (t1 == t2 == "C"):
                    new_edge = list(G.neighbors(node))
                    G.remove_node(node)
                    G.add_edge(new_edge[0], new_edge[1], type=t1)                    
                    break
        else:
            repeat = False
    return G


def internal_nodes(G: nx.Graph) -> list[tuple[int, int]]:
    """Returns all nodes except those on the boudary (ie., i, o, and g)"""
    excluded_keys = ["i", "o", "g"]
    filtered = [(key, value) for key, value in G.degree() if key not in excluded_keys]
    return filtered


def has_danglers(G: nx.Graph) -> bool:
    """Checks if graph G has any unconnected edges on its inner nodes"""
    extras = internal_nodes(G)
    for _, deg in extras:
        if deg < 2:
            return True
    return False


def permute_nodes(G: nx.Graph) -> list[nx.Graph]:
    """Generates all permutations of the nodes"""
    out: list[nx.Graph] = []
    original_nodes = list(G.nodes())
    for p in itertools.permutations(original_nodes):
        mapping = dict(zip(original_nodes, p))
        H = nx.relabel_nodes(G, mapping, copy=True)
        if not has_danglers(H):
            nx.set_node_attributes(H, attrs)
            for q in out[:]:  # need new list
                if nx.is_isomorphic(H, q, node_match=nm):
                    break
            else:
                out.append(H)
    return out

def permute_edges(G: nx.Graph) -> list[nx.Graph]:
    """Generates all edge permutations"""
    res = []
    ecombs = edge_combs(len(G.edges))
    for comb in ecombs:
        L = label_edges(G.copy(), comb)
        res.append(L)
    return res


def label_edges(G: nx.Graph, labels: tuple[str, ...], data_name: str = "type") -> nx.Graph:
    """Applies labels to each edge on data_name attribute"""
    edges = list(G.edges)
    edge_dict = dict(zip(edges, labels))
    nx.set_edge_attributes(G, edge_dict, data_name)
    return G


def edge_combs(n_edges: int) -> list[tuple[str, ...]]:
    """Generates all combinations of edge types. P represents a parallel connection
    of L & C"""
    elements = ["L", "C", "P"]
    combinations_iterator = itertools.combinations_with_replacement(elements, n_edges)
    all_combinations = list(combinations_iterator)
    return all_combinations


def most_square_grid(n: int) -> tuple[int,int]:    

    s = int(np.sqrt(n))
    candidates = []
    for w in range(max(1, s - 3), s + 4):
        h = np.ceil(n / w)
        candidates.append((abs(h - w), w*h - n, w, h))
    candidates.sort()
    _, _, w, h = candidates[0]
    return int(w), int(h)


def multidraw(g_list: list[nx.Graph], show_labels: bool = False, layout: str = "planar") -> None:
    """Function for drawing graphs. Primarily intended for drawing the atlas
    graphs. This will not work for multigraphs"""
    
    if not isinstance(g_list, list):
        g_list = [g_list]

    custom_labels = {
        "i": "$In$",
        "o": "$Out$",
        "g": "$GND$",
    }
    
    r, c = most_square_grid(len(g_list))
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(8, 8))
    
    for idx, g in enumerate(g_list):

        node_index = list(g.nodes)
        color_map = ["grey"] * len(node_index)
        color_map[node_index.index("i")] = "r"
        color_map[node_index.index("o")] = "b"
        color_map[node_index.index("g")] = "g"

        ax = plt.subplot(r, c, idx + 1)

        pos = layouts[layout](g)

        nx.draw(
            g,
            node_size=30,
            with_labels=show_labels,
            node_color=color_map,
            font_size=12,
            font_color="blue",
            pos=pos,
            labels=custom_labels,
            edge_color="tab:grey",
        )

        edge_labels = nx.get_edge_attributes(g, "type")
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="grey")

        ax.set_box_aspect(1)
    
    # Delete the unused axes
    if len(g_list) > 1:
        ax_flat = axes.ravel()
        for ax in ax_flat[len(g_list):]:
            fig.delaxes(ax)
    
    
    
# this is intended to dispaly multigraphs
def draw(G: nx.Graph, layout: str = "shell", save_file:str|None=None) -> None:
    """Function for drawing a single multigraph (i.e., a circuit)"""

    connectionstyle = [f"arc3,rad={r}" for r in itertools.accumulate([0.15] * 4)]
    custom_labels = {
        "i": "$In$",
        "o": "$Out$",
        "g": "$GND$",
    }

    node_index = list(G.nodes)
    color_map = ["grey"] * len(node_index)
    color_map[node_index.index("i")] = "r"
    color_map[node_index.index("o")] = "b"
    color_map[node_index.index("g")] = "g"

    pos = layouts[layout](G)

    #fig, ax = plt.subplots(1,1,figsize=(5,5))
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(frameon=False)
    
    nx.draw_networkx_nodes(G, pos, node_color=color_map, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="black", labels=custom_labels, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax)

    labels = {
        tuple(edge): f"{attrs['type']}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
        
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.5,
        font_color="black",
        bbox={"alpha": 0},
        ax=ax
    )
    
    
    if save_file:                
        plt.savefig(save_file, dpi=72, bbox_inches='tight')
        plt.close(fig)
    else:
        pass
        #plt.show()


def build_atlas(n_edges: int) -> list[nx.Graph]:
    """This function builds the set of non-isomorphic graphs meeting the 
    edge bounding conditions."""
    
    min_nodes, max_nodes, min_edges, max_edges = find_bounds(n_edges)
    res = []
    for i in range(1, 1253):
        g = nx.graph_atlas(i)
        if nx.is_connected(g) and min_edges <= len(g.edges) <= max_edges:
            g = nx.relabel_nodes(g, labels, copy=True)
            res.append(g)
        if len(g.nodes) > max_nodes:
            break
    return res

#def build_circuits(atlas, index, n_edges):

def build_circuits(atlas: list[nx.Graph], n_edges: int, indices: list|None = None) -> list[nx.MultiGraph]:
    """This functions expands the atlas graphs by generating all terminal 
    perturbations, edge perturbations, and component combinations """
    final = []
    
    if indices:
        atlas_subset = [atlas[i] for i in indices]
    else:
        atlas_subset = atlas
    
    for h in atlas_subset:
        perms = permute_nodes(h)
        for p in perms:
            h_e = permute_edges(p)
            h_s = [simplify(g.copy()) for g in h_e]
            h_v = [g for g in h_s if validate_edge_count(g.copy(), n_edges)]
            final.extend(h_v)

    circuits = [to_rlc(g) for g in final]
    return circuits

#%%
n = 4
atlas = build_atlas(n)
circs = build_circuits(atlas, n)
#%%
g = Graphgen(3)
g.build_all()
g.save_images("c://temp//")