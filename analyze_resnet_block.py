"""
Analyze the topology of a ResNet-18 residual block by converting it to a channel graph
and comparing structural metrics vs a randomly initialized block.
"""

import numpy as np
import torch
import networkx as nx
from torchvision.models import resnet18, ResNet18_Weights

def global_efficiency(G: nx.Graph) -> float:
    n = len(G)
    if n <= 1:
        return 0.0
    denom = n * (n - 1)
    total = 0.0
    for node in G.nodes:
        lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
        for other, d in lengths.items():
            if other == node:
                continue
            total += 1.0 / d
    return total / denom

def conv_to_graph(conv: torch.nn.Conv2d) -> nx.Graph:
    '''
    Build an undirected channel graph from a Conv2d layer.

    Nodes:
    - input channels : 0 .. C_in - 1
    - output channels: C_in .. C_in + C_out - 1

    Edges:
    - between input j and output i with weight = inverse of kernel Frobenius norm
    '''
    W = conv.weight.detach().cpu().numpy()
    C_out, C_in, kH, kW = W.shape

    # kernel strength matrix: (C_out, C_in)
    strength = np.linalg.norm(W.reshape(C_out, C_in, -1), axis=-1)

    num_nodes = C_in + C_out
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i_out in range(C_out):
        for j_in in range(C_in):
            if strength[i_out, j_in] > 1e-9:
                node_in = j_in
                node_out = C_in + i_out
                A[node_in, node_out] = 1.0 / strength[i_out, j_in]
    
    # turn into networkx graph and clean up
    G = nx.from_numpy_array(A)

    # keep only the largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G

def block_to_graph(block) -> nx.Graph:
    # Get weights for both convolutions
    W1 = block.conv1.weight.detach().cpu().numpy()
    W2 = block.conv2.weight.detach().cpu().numpy()

    # Calculate Frobenius norms after reshape (axis=-1 after reshape)
    C1_out_b, C1_in_a, _, _ = W1.shape
    C2_out_c, _, _, _ = W2.shape
    s1 = np.linalg.norm(W1.reshape(C1_out_b, C1_in_a, -1), axis=-1)
    s2 = np.linalg.norm(W2.reshape(C2_out_c, C1_out_b, -1), axis=-1)

    # Define node offsets
    num_nodes = C1_in_a + C1_out_b + C2_out_c

    # Make adjacency matrix
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # a -> b
    for i_out in range(C1_out_b):
        for j_in in range(C1_in_a):
            if s1[i_out, j_in] > 1e-9:
                node_in = j_in
                node_out = C1_in_a + i_out
                A[node_in, node_out] = 1.0 / s1[i_out, j_in]
    # b -> c
    for i_out in range(C2_out_c):
        for j_in in range(C1_out_b):
            if s2[i_out, j_in] > 1e-9:
                node_in = C1_in_a + j_in
                node_out = C1_in_a + C1_out_b + i_out
                A[node_in, node_out] = 1 / s2[i_out, j_in]
    # a -> c (skip connections)
    if block.downsample is not None:
        # kernel
        skip = block.downsample[0].weight.detach().cpu().numpy()
        ss = np.linalg.norm(skip.reshape(C2_out_c, C1_in_a, -1), axis=-1)
        
        for i_in in range(C1_in_a):
            for j_out in range(C2_out_c):
                if ss[j_out, i_in] > 1e-9:
                    node_in = i_in
                    node_out = C1_in_a + C1_out_b + j_out
                    A[node_in, node_out] = 1 / ss[j_out, i_in]
    else:
        # identity skip
        for i in range(C2_out_c):
            node_in = i
            node_out = C1_in_a + C1_out_b + i
            A[node_in, node_out] = 1.0
    
    # turn into network graph
    G = nx.from_numpy_array(A)

    return G  

def graph_metrics(G: nx.Graph, G_ref: nx.Graph | None = None) -> dict:
    if not nx.is_connected(G.to_undirected()):
        print("WARNING: Graph is disconnected. Analyzing largest connected component.")
        largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
        G = G.subgraph(largest_cc).copy()

    N = G.number_of_nodes()
    m = G.number_of_edges()
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G, weight='weight')
    E = global_efficiency(G)

    metrics = {
        "N": N,
        "m": m,
        "C": C,
        "L": L,
        "E": E,
    }

    if G_ref is not None:
        if not nx.is_connected(G_ref.to_undirected()):
            print("WARNING: Reference graph is disconnected. Analyzing largest connected component.")
            largest_cc = max(nx.connected_components(G_ref.to_undirected()), key=len)
            G_ref = G_ref.subgraph(largest_cc).copy()
        
        N_ref = G_ref.number_of_nodes()
        m_ref = G_ref.number_of_edges()
        C_ref = nx.average_clustering(G_ref)
        L_ref = nx.average_shortest_path_length(G_ref, weight='weight')
        E_ref = global_efficiency(G_ref)

        # small-worldness "relative to init" index
        sigma = (C / C_ref) / (L / L_ref)

        # integration efficiency
        IE = E / m
        IE_ref = E_ref / m_ref
        IE_norm = IE / IE_ref

        metrics.update({
            "N_ref": N_ref,
            "m_ref": m_ref,
            "C_ref": C_ref,
            "L_ref": L_ref,
            "E_ref": E_ref,
            "sigma": sigma,
            "IE": IE,
            "IE_ref": IE_ref,
            "IE_norm": IE_norm,
        })

    return metrics

def main():
    # load pre-trained ResNet-18
    weights = ResNet18_Weights.IMAGENET1K_V1
    model_trained = resnet18(weights=weights)
    model_trained.eval()

    # random init with ResNet-18 architecture
    model_init = resnet18(weights=None)
    model_init.eval()

    # pick same block in both and make graphs
    G_trained = block_to_graph(model_trained.layer1[0])
    G_init = block_to_graph(model_init.layer1[0])

    metrics = graph_metrics(G_trained, G_init)

    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()