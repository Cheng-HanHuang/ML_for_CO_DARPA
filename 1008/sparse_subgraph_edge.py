import torch
import networkx as nx
import numpy as np
import psutil, os


def build_upper_edge_list_from_graph(G: nx.Graph):
    """
    Store the edges of the undirected graph G as an edge list in upper triangular form (u < v).

    U, V: 1D torch.long tensors (length = number of edges |E|),
    representing the two endpoint indices (u, v) for each edge, where u < v.
    """
    edges = []
    for u, v in G.edges():
        if u == v: # Exclude self-loops
            continue
        if u > v:
            u, v = v, u
        edges.append((u, v))

    if not edges:
        # The case with no edges
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    U = torch.tensor([e[0] for e in edges], dtype=torch.long)
    V = torch.tensor([e[1] for e in edges], dtype=torch.long)
    return U, V


def sample_edge_sublist(U: torch.Tensor, V: torch.Tensor, p_keep: float):
    """
    Perform Bernoulli sampling for each "undirected edge" (u<v) to keep it with probability p_keep.
    Return the selected sub-edge list U_sub, V_sub (still in the u<v upper triangular form).
    """
    assert U.dtype == torch.long and V.dtype == torch.long
    assert U.shape == V.shape

    m = U.numel()
    mask = torch.rand(m) < p_keep
    idx = mask.nonzero(as_tuple=False).squeeze(1)

    if idx.numel() == 0:
        # When no edges are sampled, return an empty sublist.
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    return U.index_select(0, idx), V.index_select(0, idx)


def matrix_free_spmv(U_sub, V_sub, x, n, p_keep):
    """
    Matrix-Free Sparse Matrix-Vector Product
    y = (A_sub / p_keep) @ x
    """
    if U_sub.numel() == 0:
        return torch.zeros(n, dtype=x.dtype)

    y = torch.zeros(n, dtype=x.dtype)
    x_v = x.index_select(0, V_sub)
    x_u = x.index_select(0, U_sub)

    # y[u] += x[v] and y[v] += x[u]
    y.scatter_add_(0, U_sub, x_v)
    y.scatter_add_(0, V_sub, x_u)

    return y / p_keep


def check_maximal_independent_set(X_torch, n, edges_g_set):
    """
    Check whether a solution is a Maximal Independent Set based on an edge list (hash set).
    """
    # 1. Retrieve the candidate node set
    X_binarized = X_torch.bool()
    S_nodes_tensor = X_binarized.nonzero(as_tuple=False).squeeze(1)
    S_nodes_set = set(S_nodes_tensor.tolist())

    # 2. Check whether S is an independent set:
    # iterate through all node pairs in S, and if any pair is found in the edge hash set, then S is not an IS
    for u in S_nodes_tensor:
        for v in S_nodes_tensor:
            if u >= v: continue
            if (u.item(), v.item()) in edges_g_set:
                return False, None

    # 3. Check whether S is a maximal independent set:
    # iterate through all nodes w not in S, and check whether w can be added to S
    all_nodes = set(range(n))
    outside_nodes = all_nodes - S_nodes_set

    for w in outside_nodes:
        can_be_added = True
        for s_node in S_nodes_set:
            u, v = min(w, s_node), max(w, s_node)
            if (u, v) in edges_g_set:
                can_be_added = False
                break
        if can_be_added:
            # If there exists a node that can be added, then S is not maximal
            return False, None

    # If all checks are passed, then S is a Maximal Independent Set
    return True, S_nodes_tensor

#########################################################################
# Begin
#########################################################################
node_list = [200, 400, 600, 800]
#node_list = [200, 400]
p = 0.5

p_keep = 1.0

with open("MaxIS_edge_random_subgraph.txt", "w") as f:
    for n in node_list:
        best_list = []
        conv_iters_all = []
        for seed in range(50):
            print(f"\n===== Running for n = {n}, seed = {seed} =====")
            f.write(f"Number of nodes n = {n}\n")
            f.write(f"Edge probability p = {p}\n")
            f.write(f"Graph Seed = {seed}\n")
            f.write("Results for each epoch:\n")

            G = nx.gnp_random_graph(n, p, seed=seed)

            # Completely remove all adjacency matrix constructions and use only edge lists.
            print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            U, V = build_upper_edge_list_from_graph(G)

            # Precompute a hash set of edges for efficient lookup in the MIS checker
            edges_g_set = set(zip(U.tolist(), V.tolist()))

            # Construct the edge list of the complement graph, then discard the complement graph object
            complement_G = nx.complement(G)
            U_comp, V_comp = build_upper_edge_list_from_graph(complement_G)
            del complement_G  # Free up memory

            # --------------------------------------------------------------------------
            ### Here, we set gamma, gamma', and beta.
            gamma_c = 1

            # Compute the degree in the complement graph
            degrees_g = dict(G.degree())
            degrees_c = {node: (n - 1) - degree for node, degree in degrees_g.items()}
            max_degree = max(degrees_c.values()) if degrees_c else 0
            gamma = 2 + max_degree
            # --------------------------------------------------------------------------

            beta = 0.8
            alpha = 0.0001
            ITERATION_T = 150
            epoch = 50

            # Degree-based initialization
            degrees = dict(G.degree())
            max_degree_node = max(degrees.values()) if degrees else 1
            d = torch.zeros(n)
            for node, degree in G.degree():
                d[node] = 1 - (degree / max_degree_node)
            if max(d) > 0: d = d / max(d)

            input_velocity = torch.zeros(n)
            exploration_parameter_eta = 2.0
            covariance_matrix = exploration_parameter_eta * torch.eye(len(d))
            best_MIS_size = 0
            best_epoch = -1
            best_iter = -1
            conv_iters_seed = []

            for init in range(epoch):
                if init == 0:
                    input_tensor = d.clone()
                else:
                    torch.manual_seed(init + seed)  # Ensure that each random initialization is different
                    input_tensor = torch.normal(mean=d, std=torch.sqrt(torch.diag(covariance_matrix)))

                input_velocity = torch.zeros(n)
                converged = False
                conv_iter = None
                MIS_size = 0

                for iteration in range(ITERATION_T):
                    # 1. Perform random edge sampling within each iteration
                    U_sub, V_sub = sample_edge_sublist(U, V, p_keep)
                    # U_comp_sub, V_comp_sub = sample_edge_sublist(U_comp, V_comp, p_keep)

                    # 2. Compute gradient components using a matrix-free method
                    ax = matrix_free_spmv(U_sub, V_sub, input_tensor, n, p_keep)
                    # acx = matrix_free_spmv(U_comp_sub, V_comp_sub, input_tensor, n, p_keep)

                    # 3. Synthesize the final gradient
                    # gradient = -torch.ones(n) + (gamma * ax - gamma_c * acx)
                    gradient = -torch.ones(n) + gamma * ax
                    # ----------------------------------------

                    input_velocity = beta * input_velocity + alpha * gradient
                    input_tensor = torch.clamp(input_tensor - input_velocity, 0, 1)

                    # Call the new edge-list-based MIS checker
                    Checker, MIS = check_maximal_independent_set(input_tensor, n, edges_g_set)

                    if Checker:
                        MIS_size = MIS.numel()
                        process = psutil.Process(os.getpid())
                        cpu_mem = process.memory_info().rss / 1024 / 1024
                        conv_iter = iteration
                        converged = True
                        break

                if converged:
                    conv_iters_seed.append(conv_iter)
                    process = psutil.Process(os.getpid())
                    cpu_mem = process.memory_info().rss / 1024 / 1024
                    f.write(
                        f"Epoch {init}, MIS size = {MIS_size}, converged at iteration = {conv_iter}, CPU memory = {cpu_mem:.2f} MB\n"
                    )
                    if MIS_size > best_MIS_size:
                        best_MIS_size = MIS_size
                        best_epoch = init
                        best_iter = conv_iter
                else:
                    f.write(f"Epoch {init}, No convergence (ran {ITERATION_T} iterations)\n")

            best_list.append(best_MIS_size)
            if conv_iters_seed:
                avg_conv_iter = np.mean(conv_iters_seed)
                conv_iters_all.extend(conv_iters_seed)
                f.write(f"\nAverage convergence iteration (this seed) = {avg_conv_iter:.2f}\n")
            else:
                f.write("\nNo epoch converged in this seed.\n")
            f.write(f"\nBest MIS size = {best_MIS_size}\n")
            f.write(f"\n=============================================\n")
            print(f"n={n}, seed={seed}, best MIS size = {best_MIS_size}")

        avg_best = np.mean(best_list) if best_list else 0
        if conv_iters_all:
            avg_conv_all = np.mean(conv_iters_all)
            f.write(
                f"\n=**********************************************************************************************************\n")
            f.write(f"Average Best MIS size over {len(best_list)} seeds = {avg_best:.2f}\n")
            f.write(f"Average convergence iteration over all converged epochs = {avg_conv_all:.2f}\n")
            f.write(
                f"=************************************************************************************************************\n\n")
        else:
            f.write("\nNo convergence across all seeds.\n")