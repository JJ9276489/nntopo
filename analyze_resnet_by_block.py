import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

# Helper functions

def global_efficiency(G: nx.Graph) -> float:
    '''Calculates global efficiency (E) for an undirected graph'''
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

def block_to_graph(block) -> nx.Graph:
    '''
    Converts a ResNet BasicBlock into an unidrected graph.
    Nodes: [Inputs] + [Internal] + [Outputs]
    Edges: Weighted by inverse Frobenius norm of kernals.
    '''

    # extract weights
    W1 = block.conv1.weight.detach().cpu().numpy()
    W2 = block.conv2.weight.detach().cpu().numpy()

    # calculate norms
    C1_out_b, C1_in_a, _, _= W1.shape
    C2_out_c, _, _, _ = W2.shape

    # norm shape: (out, in)
    s1 = np.linalg.norm(W1.reshape(C1_out_b, C1_in_a, -1), axis=-1)
    s2 = np.linalg.norm(W2.reshape(C2_out_c, C1_out_b, -1), axis=-1)

    # initialize adjacency matrix
    num_nodes = C1_in_a + C1_out_b + C2_out_c

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # a -> b
    mask1 = s1 > 1e-9
    valid_s1 = s1.T
    valid_mask = mask1.T

    block_ab = A[0:C1_in_a, C1_in_a:C1_in_a + C1_out_b]
    np.divide(1.0, valid_s1, out=block_ab, where=valid_mask)

    # b -> c
    mask2 = s2 > 1e-9
    valid_s2 = s2.T
    valid_mask2 = mask2.T

    block_bc = A[C1_in_a:C1_in_a + C1_out_b, C1_in_a + C1_out_b:]
    np.divide(1.0, valid_s2, out=block_bc, where=valid_mask2)

    # a -> c
    if block.downsample is not None:
        skip = block.downsample[0].weight.detach().cpu().numpy()
        ss = np.linalg.norm(skip.reshape(C2_out_c, C1_in_a, -1), axis=-1)

        mask_s = ss > 1e-9
        valid_ss = ss.T
        valid_mask_s = mask_s.T

        block_ac = A[0:C1_in_a, C1_in_a + C1_out_b:]
        np.divide(1.0, valid_ss, out=block_ac, where=valid_mask_s)
    else:
        limit = min(C1_in_a, C2_out_c)
        if C1_in_a != C2_out_c:
            print(f"Warning: Identity skip dim mismatch {C1_in_a} vs {C2_out_c}")
        
        indices = np.arange(limit)
        A[indices, C1_in_a + C1_out_b + indices] = 1.0
    
    G = nx.from_numpy_array(A)    # undirected by default

    return G

def get_sigma_E(G: nx.Graph, G_ref: nx.Graph):
    '''Calculates small worldedness using efficiency: (C/C_rand) / (E_rand/E)'''

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if not nx.is_connected(G_ref):
        G_ref = G_ref.subgraph(max(nx.connected_components(G_ref))).copy()

    C = nx.average_clustering(G)
    E = global_efficiency(G)

    C_ref = nx.average_clustering(G_ref)
    E_ref = global_efficiency(G_ref)

    if C_ref < 1e-9 or E < 1e-9:
        return 0.0
    
    return (C / C_ref) / (E_ref / E)

def get_IE_norm(G: nx.Graph, G_ref: nx.Graph):
    '''Calculates integration efficiency normalized by random graph'''
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if not nx.is_connected(G_ref):
        G_ref = G_ref.subgraph(max(nx.connected_components(G_ref))).copy()
    
    E = global_efficiency(G)
    m = G.number_of_edges()

    E_ref = global_efficiency(G_ref)
    m_ref = G_ref.number_of_edges()

    IE = E / m if m > 0 else 0
    IE_ref = E_ref / m_ref if m_ref > 0 else 0

    return IE / IE_ref if IE_ref > 1e-9 else 0

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading Models...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model_trained = resnet18(weights=weights).eval()
    model_init = resnet18(weights=None).eval()

    layers = ['layer1', 'layer2', 'layer3', 'layer4']

    results = {
        'labels': [],
        'sigma': [],
        'ie': []
    }

    print("Starting topological analysis...")
    print("-" * 60)
    print(f"{'Block':<15} | {'Sigma':<10} | {'IE_Norm':<10}")
    print("-" * 60)

    for layer_name in layers:
        layer_t = getattr(model_trained, layer_name)
        layer_i = getattr(model_init, layer_name)

        for b_idx, (block_t, block_i) in enumerate(zip(layer_t, layer_i)):
            label = f"{layer_name}.{b_idx}"

            G_t = block_to_graph(block_t)
            G_i = block_to_graph(block_i)

            s = get_sigma_E(G_t, G_i)
            ie = get_IE_norm(G_t, G_i)

            results['labels'].append(label)
            results['sigma'].append(s)
            results['ie'].append(ie)

            print(f"{label:<15} | {s:.4f}     | {ie:.4f}")
        
    x = range(len((results['labels'])))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = '#1f77b4' # Muted Blue for sigma
    ax1.set_xlabel('Network Depth (Block)', fontsize=12)
    ax1.set_ylabel('Sigma_E (Small-Worldness)', color=color, fontsize=12)
    ax1.plot(x, results['sigma'], color=color, marker='o', linewidth=2, label='Small-Worldness')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)    

    ax2 = ax1.twinx()  
    color = '#ff7f0e' # Muted Orange for IE
    ax2.set_ylabel('Normalized Integration Efficiency', color=color, fontsize=12)
    ax2.plot(x, results['ie'], color=color, marker='s', linestyle='--', linewidth=2, label='Integration Eff.')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Topological Maturation of ResNet-18', fontsize=14, pad=15)
    plt.xticks(x, results['labels'], rotation=45)

    plt.tight_layout()
    plt.savefig('resnet_maturation.png', dpi=300, bbox_inches='tight')
    print("-" * 60)
    print("Plot saved to 'resnet_maturation.png'")
    plt.show()

if __name__ == "__main__":
    main()