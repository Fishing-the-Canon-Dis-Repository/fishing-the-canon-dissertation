#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A Candidate for the MSc in Digital Scholarship 2024-25, University of Oxford 

NB: ChatGPT was used to "vibe-code" much of the contents of this notebook and debug manually created sections.
"""

#%% Importing Packages
import pandas as pd # version 2.2.3
import scipy.stats as st # version 1.16.0
import scipy.spatial as sp # version 1.16.0
from scipy.sparse import issparse # version 1.16.0
from scipy.optimize import fsolve # version 1.16.0
import networkx as nx # version 3.4.2
import numpy as np # version 2.3.0
from numpy.linalg import eig, norm # version 2.3.0
import igraph as ig # version 0.11.9
import tqdm # version 4.67.1
from sklearn.mixture import GaussianMixture # version 1.7.1
import matplotlib.pyplot as plt # version 3.10.3
from sklearn.preprocessing import StandardScaler # version 1.7.1
from scipy.spatial.distance import pdist, squareform # version 1.16.0
# Importing Modules from Python 3.12.11
import itertools as it
import ast
import math
import pygmmis
import time
from collections import Counter
import re
import matplotlib
import collections

#%% Loading dataframes

vector_data = pd.read_csv("FILEPATH/BookVectorData_withIDs.csv")

TM_data = pd.read_csv("FILEPATH//manual_title_subset.tsv",
                      sep="\t")

#%% Removing texts with meta content 

TM_data = TM_data.set_index("docid") # Setting index with text ID

vector_data = vector_data.set_index("FileName") # Setting index with text ID

vector_data = vector_data.drop("volume-rights.txt")
vector_data = vector_data.drop("miun.aje0708,0011,001.txt")

#%% Calculating Percentatge of Texts from English Speaking Countries & Other Corpus Data

print(len(vector_data.index))

nationalities = {}
for country in list(i.strip() for i in pd.unique(TM_data["nationality"]) if isinstance(i, str)):
    counter = 0 
    for i in vector_data.index:
        if TM_data.at[i.removesuffix(".txt"), "nationality"] == country: 
            counter += 1
    nationalities[country] = counter


sum(list(nationalities[i] for i in ('uk', 'us', 'ir', 'ca')))
#1426

in_copyright = []
for i in vector_data.index: 
    if int(TM_data.at[i.removesuffix('.txt'), "firstpub"]) >= 1920:
        in_copyright.append(i)

print(len(in_copyright)) #836
print(836/2016) # 0.415

sum(vector_data["TotalBookTokens"]) / 2016 # 120685.878

auth_genders = {}
for gender in list(i.strip() for i in pd.unique(TM_data["gender"]) if isinstance(i,str)):
    count = 0
    for i in vector_data.index:
        if TM_data.at[i.removesuffix('.txt'), "gender"] == gender: 
            count += 1
    auth_genders[gender] = count / 2016 if count != 0 else 0

print(auth_genders) # 'u': 0.07192460317460317, 'm': 0.6011904761904762, 'f': 0.32688492063492064}

#%% Calculating Distance Distributions

# Defining cosine distance function 
def get_cos_distances(embeddings): 
    Cos_distances = {}
    for pair in it.combinations(embeddings.items(),2):
        x_label, x = pair[0]
        y_label, y = pair[1]
        dist = sp.distance.cosine(x,y)
        Cos_distances.update({f"{x_label.removesuffix(".txt")} - {y_label.removesuffix(".txt")}": dist})
    return Cos_distances

# Defining a dictionary of vectors 
vectors = {}
for i in vector_data.index: 
    vectors[i.removesuffix(".txt")] = np.array(ast.literal_eval(vector_data.at[i, "BookVector"]))
    
# Defining a dictionary of distances
distances = get_cos_distances(vectors)

#%% 
""" 
The following tests the bimodality of the distribution of all distances between vectors.
This is meant to establish the nature of the distrubtions and prepare for a datadrive approach to threshold setting.
"""

dists = np.array([d for d in distances.values()]) # Converting distances in dictionary to array


# === Parameters ===
EPSILON = 1e-6  # to prevent 0 or 1 in transformation

# === Step 1: Logit transform ===
def arcsin_sqrt(x):
    return np.arcsin(np.sqrt(np.clip(x, 0, 1)))

def inverse_arcsin_sqrt(y):
    return (np.sin(y))**2

# === Step 2: Fit GMM in transformed space ===
def fit_arcsin_sqrt_gmm(data, n_components=2):
    transformed = arcsin_sqrt(data).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(transformed)
    return gmm

# === Step 3: Find intersection of component PDFs ===
def find_intersection(gmm):
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    def eq(x):
        return weights[0] * norm.pdf(x, means[0], stds[0]) - weights[1] * norm.pdf(x, means[1], stds[1])

    # Use midpoint between means as initial guess
    guess = np.mean(means)
    intersect_logit = fsolve(eq, guess)[0]
    return intersect_logit, inverse_arcsin_sqrt(intersect_logit)

# === Step 4: Visualization (Optional) ===
def plot_fit(data, gmm, intersection_logit):
    x_vals = np.linspace(EPSILON, 1 - EPSILON, 1000)
    logit_x = arcsin_sqrt(x_vals).reshape(-1, 1)

    # Evaluate weighted sum of component PDFs
    pdf_vals = np.exp(gmm.score_samples(logit_x))

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, density=True, alpha=0.5, label='Empirical')
    plt.plot(x_vals, pdf_vals, label='GMM (transformed)', lw=2)
    plt.axvline(x=inverse_arcsin_sqrt(intersection_logit), color='r', linestyle='--', label='Intersection')
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title("Arcsine/Square Root-GMM Fit to Cosine Distances")
    plt.legend()
    plt.show()

# === Implementation ===
gmm = fit_arcsin_sqrt_gmm(dists, n_components=2)
intersect_logit, intersect_original = find_intersection(gmm)

print(f"Intersection point in logit space: {intersect_logit:.4f}")
print(f"Intersection epoint in original space: {intersect_original:.4f}")

plot_fit(dists, gmm, intersect_logit)

for k in range(1, 5):
    gmm = GaussianMixture(n_components=k, random_state=0).fit(arcsin_sqrt(dists).reshape(-1, 1))
    print(f"k={k} | AIC={gmm.aic(arcsin_sqrt(dists).reshape(-1,1)):.2f} | BIC={gmm.bic(arcsin_sqrt(dists).reshape(-1,1)):.2f}")

#%% Constructing a graph based on the intersection of the two points 

graph = nx.DiGraph()

graph_weighted = nx.DiGraph()

# Constructeding the thresholded graph
graph.add_nodes_from(vectors.keys())

for i, d in distances.items(): 
    
    if d > intersect_original: 
        continue
    novel1, sep, novel2 = i.partition(" - ")
    year1 = TM_data.at[novel1, "firstpub"]
    year2 = TM_data.at[novel2, "firstpub"]
    if year1 == year2:
        graph.add_edge(novel1, novel2)
        graph.add_edge(novel2, novel1)
    elif year1 < year2: 
        graph.add_edge(novel1, novel2)
    elif year1 > year2: 
        graph.add_edge(novel2, novel2)
        
# Construct the weighted graph

graph_weighted.add_nodes_from(vectors.keys())

for i, d in distances.items(): 
    novel1, sep, novel2 = i.partition(" - ")
    year1 = TM_data.at[novel1, "firstpub"]
    year2 = TM_data.at[novel2, "firstpub"]
    weight = 1 - d
    if year1 == year2:
        graph_weighted.add_edge(novel1, novel2, weight=weight)
        graph_weighted.add_edge(novel2, novel1, weight=weight)
    elif year1 < year2: 
        graph_weighted.add_edge(novel1, novel2, weight=weight)
    elif year1 > year2: 
        graph_weighted.add_edge(novel2, novel2, weight=weight)
        
#%% Calculating & Comparing betweenness centrality of both graphs

betweenness = nx.betweenness_centrality(graph, weight=None)

betweenness_weighted = nx.betweenness_centrality(graph_weighted, weight='weight')

top_50 = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:50]
top_50_weighted = sorted(betweenness_weighted.items(), key=lambda x: x[1], reverse=True)[:50]

jaccard = len(top_50 & top_50_weighted) / len(top_50 | top_50_weighted)

print(top_50)
print(top_50_weighted)

from scipy.stats import spearmanr

rho, p = spearmanr([v for i, v in betweenness.items()],[v for i, v in betweenness_weighted.items()])

rho, p = spearmanr([v for i,v in top_50], [v for i,v in top_50_weighted])
print(rho, p)

#%% Setting up the longterm cost function for Scholkemper & Schaub's Equitable Partition Optimization 

"""
Here we set up the code to evaluate the long term cost function that Scholkemper & Schaub proposed.

"""

def compute_dominant_eigenvalue(A):
    """Return spectral radius of A."""
    eigvals = eig(A)[0]
    return np.max(np.abs(eigvals))

def generate_soft_assignment(n, k, method="random", seed=None):
    """Generate a soft, row-stochastic assignment matrix H."""
    if seed is not None:
        np.random.seed(seed)
    H = np.random.rand(n, k)
    H /= H.sum(axis=1, keepdims=True)
    return H

def compute_true_long_term_cost(A, H, tol=1e-6, verbose=False, max_iter=200):
    """Compute true long-term cost Γ_{∞-EP} as defined by infinite series until convergence."""
    n, k = H.shape
    D_inv = np.diag(1 / H.sum(axis=0))
    projector = H @ D_inv @ H.T
    rho = compute_dominant_eigenvalue(A)
    A_scaled = A / rho
    At = np.eye(n)
    total_cost = 0.0
    delta = np.inf
    t = 1
    if verbose:
        print("  Computing terms in Γ_{∞-EP} series...")
    while delta > tol and t <= max_iter:
        if verbose:
            print(f"    t = {t}, last term = {delta:.2e}")
        At = At @ A_scaled
        diff = At @ H - projector @ At @ H
        term = norm(diff, 'fro') ** 2 / (rho ** t)
        total_cost += term
        delta = term
        t += 1
    return total_cost

def evaluate_true_costs_over_k(A, k_values, trials=5):
    """Evaluate true long-term cost over range of k using multiple initializations."""
    n = A.shape[0]
    costs = []
    print("Evaluating long-term costs over k...")
    for k in tqdm.tqdm(k_values, desc="k sweep"):
        best_cost = np.inf
        for _ in range(trials):
            H = generate_soft_assignment(n, k, seed=np.random.randint(1e9))
            cost = compute_true_long_term_cost(A, H)
            if cost < best_cost:
                best_cost = cost
        costs.append(best_cost)
    return np.array(costs)

def convert_to_dense_if_needed(A):
    """Ensure the matrix A is a dense NumPy array."""
    if issparse(A):
        return A.toarray()
    elif isinstance(A, np.ndarray):
        return A
    else:
        raise TypeError("Input must be a NumPy array or SciPy sparse matrix.")


def ensure_square_matrix(A):
    """Ensure matrix A is square, or raise an informative error."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency matrix must be square. Got shape {A.shape}.")
    return A


def find_knee_point(costs, k_values):
    """Find knee point using maximum distance to chord method."""
    x = np.array(k_values)
    y = costs
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    line = y_norm[0] + (y_norm[-1] - y_norm[0]) * (x_norm - x_norm[0]) / (x_norm[-1] - x_norm[0])
    distances = np.abs(y_norm - line)
    return x[np.argmax(distances)]

k_values = [k for k in range(2,21)]

lt_knee_weighted = find_knee_point(evaluate_true_costs_over_k(ensure_square_matrix(convert_to_dense_if_needed(nx.adjacency_matrix(graph_weighted))), k_values), k_values)
# Was 6 

lt_knee_unweighted = find_knee_point(evaluate_true_costs_over_k(ensure_square_matrix(convert_to_dense_if_needed(nx.adjacency_matrix(graph))), k_values), k_values)
# was 6
print(lt_knee_weighted)
print(lt_knee_unweighted)

#%%
"""
We now use the knee points to (both of which were 6) to generate the partition. 
Specifically, we will use a binary partition that does soft assignment and stochastically models the assigment of each node. 
We will then record a binaritization of the the soft assignment to determine a node's "majority" category. 
The stochastic assigment will be preserved for potential futher analysis.
"""

def optimize_soft_partition(A, k, trials=10, verbose=False):
    """Return H that minimizes the true long-term cost for a given k."""
    n = A.shape[0]
    best_cost = np.inf
    best_H = None
    for _ in tqdm.tqdm(range(trials), desc="Trial"):
        H = generate_soft_assignment(n, k)
        cost = compute_true_long_term_cost(A, H, verbose=verbose)
        if cost < best_cost:
            best_cost = cost
            best_H = H
    return best_H


def save_partition_results(H, output_csv, node_ids=None, node_index=None):
    """Save stochastic and majority role assignment to CSV."""
    n, k = H.shape
    if node_ids is None:
        node_ids = list(range(n))
    if node_index is None:
        node_index = {nid: i for i, nid in enumerate(node_ids)}
    role_probs = [dict(enumerate(row)) for row in H]
    majority_roles = np.argmax(H, axis=1)
    index_positions = [node_index[nid] for nid in node_ids]
    df = pd.DataFrame({
        'node_id': node_ids,
        'role_probs': role_probs,
        'majority_role': majority_roles,
        'node_index': index_positions
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved role assignments to {output_csv}")


def graph_to_adjacency_matrix(G):
    """Convert NetworkX graph to NumPy adjacency matrix, preserving node order."""
    node_list = list(G.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}
    A = nx.to_numpy_array(G, nodelist=node_list, weight='weight', dtype=float)
    return A, node_list, node_index


def run_ep_partition_from_graph(G, k, output_csv, trials=10, verbose=False):
    """
    Run long-term EP optimization on a NetworkX graph G,
    save role assignment CSV with correct node IDs.
    """
    A, node_ids, node_index = graph_to_adjacency_matrix(G)
    H_opt = optimize_soft_partition(A, k, trials=trials, verbose=verbose)
    save_partition_results(H_opt, output_csv, node_ids=node_ids, node_index=node_index)
    return H_opt


# Creating a weighted partition and saving it as a csv file. 
H_weighted = run_ep_partition_from_graph(graph_weighted, 6, "weighted_EP6.csv", trials=100, verbose=True)

# Creating an unweighted partition and saving it as a csv file. 
H_unweighted = run_ep_partition_from_graph(graph, 6, "unweighted_EP6.csv", trials=100, verbose=True)

#%% Checking agreement between weighted and unweighted node sorting. 

weighted_df = pd.read_csv("weighted_EP6.csv")
unweighted_df = pd.read_csv("unweighted_EP6.csv")

def overlap_coefficent(weighted_df, unweighted_df, k):
    overlaps = {}
    perms = [p for p in it.permutations(range(k))]
    for per in tqdm.tqdm(perms, desc = "perm sweep"): 
        total = 0 
        for i in range(k):
            weighted_set = set(node for node in weighted_df.index if weighted_df.at[node, "majority_role"] == i)
            unweighted_set = set(node for node in unweighted_df.index if unweighted_df.at[node, "majority_role"] == per[i])
            numerator = len(unweighted_set & weighted_set)
            denominator = len(weighted_set) if len(weighted_set) > 0 else 1
            overlap = numerator / denominator 
            total += overlap 
        overlaps[per] = total
    best_perm = max(overlaps, key=overlaps.get)
    return overlaps[best_perm], best_perm
            
            
overlap_coeff, best_perm = overlap_coefficent(weighted_df, unweighted_df, 6)         
print(overlap_coeff, best_perm)
print(overlap_coeff/6)      

#%%

"""
The following will calculate the eigencentrality, betweenness centrality, closeness centrality, and PageRank.
I will then calculate the mean and the average deviation from that mean for each cluster to determine how characteristic the metric is of the cluster. 
"""



import numpy as np
import networkx as nx

def group_eigencentrlity(
    graph,
    dataframe,
    k,
    *,
    weight="weight",
    mode="in",          # "in": A.T v = λ v  (incoming prestige)
                        # "out": A   v = λ v  (outgoing influence)
    eig_tol=1e-10,      # tolerance to treat eigenvalues as real
    clip_tol=1e-12,     # clip tiny negatives (numerical fuzz)
    normalize=True      # L1-normalize the vector (optional)
):
    """
    Returns
    -------
    node_ec: dict[node -> float]
    group_ecs: dict[group -> float]      (mean EC per group)
    ec_avg_std: dict[group -> float]     (std of EC within group)
    """
    nodes = list(graph.nodes())
    A = nx.to_numpy_array(graph, weight=weight, nodelist=nodes)
    M = A.T if mode == "in" else A

    # --- eigenpair selection: largest REAL eigenvalue (Newman-style) ---
    eigvals, eigvecs = np.linalg.eig(M)

    # mask eigenvalues that are (numerically) real
    is_real = np.abs(np.imag(eigvals)) < eig_tol
    if not np.any(is_real):
        print("[WARN] No eigenvalues with negligible imaginary part found; "
              "proceeding with the eigenvalue of largest real part.")
        idx = int(np.argmax(np.real(eigvals)))
    else:
        real_vals = np.real(eigvals[is_real])
        # index within the 'real' subset having the largest real part
        local_idx = int(np.argmax(real_vals))
        # map back to the original index space
        idx = int(np.nonzero(is_real)[0][local_idx])

    v = np.real(eigvecs[:, idx])  # drop tiny imaginary parts

    # --- orientation & clipping ---
    # orient to have nonnegative average (sign of eigenvector is arbitrary)
    if v.mean() < 0:
        v = -v

    # warn if there are materially negative components before clipping
    neg_mask = v < -clip_tol
    if np.any(neg_mask):
        frac_neg = float(np.mean(neg_mask))
        min_val = float(v[neg_mask].min())
        print(f"[WARN] Selected eigenvector has {frac_neg:.4%} negative components "
              f"(min={min_val:.3e}) before clipping. Verify graph assumptions.")

    # clip tiny negatives due to rounding
    v[np.abs(v) < clip_tol] = 0.0
    v[v < 0] = 0.0

    # optional normalization (L1)
    if normalize:
        s = v.sum()
        if s > 0:
            v = v / s

    node_ec = {node: float(val) for node, val in zip(nodes, v)}

    # --- per-group aggregation ---
    df = dataframe.set_index("node_id", inplace=False)
    group_ecs, ec_avg_std = {}, {}
    for g in range(k):
        ecs = [node_ec[n] for n in nodes if (n in df.index and df.at[n, "majority_role"] == g)]
        if len(ecs) == 0:
            print(f"[WARN] Group {g} has zero members (or missing IDs in DataFrame).")
            group_ecs[g] = float("nan")
            ec_avg_std[g] = float("nan")
            continue
        group_ecs[g] = float(np.mean(ecs))
        ec_avg_std[g] = float(np.std(ecs))

    return node_ec, group_ecs, ec_avg_std

weighted_ecs, weighted_group_ecs, weighted_ec_std = group_eigencentrlity(graph_weighted, weighted_df, 6, mode='out')

print(weighted_group_ecs)    
print(weighted_ec_std)

def group_betweenness(graph, dataframe, k):
    """
    Parameters 
    Parameters
    ----------
    graph: a networkx.Graph or networkx.DiGraph
    dataframe : Pandas.DataFrame
        A dataframe where each node's group assigment is stored.
    k : int
        the number of groups.

    Returns
    -------
    dictionary of node IDs and betweenness centerlity scores, dictionary of average centeraliteis per group, dictionary of average deviation for each group. 
        
    """
    dataframe = dataframe.set_index("node_id")
    node_betweenness = nx.betweenness_centrality(graph, weight='weight')
    group_betweenness = {}
    group_betweenness_std = {}
    for n in range(k):
        scores = []
        for node in node_betweenness.keys():
            if dataframe.at[node, "majority_role"] == n: 
                scores.append(node_betweenness[node])
        avg = sum(scores) / len(scores) if len(scores) > 0 else print(f"Error! group {n} is showing a score list with length <= 0")
        group_betweenness[n] = avg
        group_betweenness_std[n] = np.std(scores)
    return node_betweenness, group_betweenness, group_betweenness_std
        
weighted_betweenness, weighted_group_betweenness, weighted_group_betweenness_std = group_betweenness(graph_weighted, weighted_df, 6)
print(weighted_group_betweenness)
print(weighted_group_betweenness_std)

def group_closeness(graph, dataframe, k):
    """

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        
    dataframe : pandas.DataFrame
        A dataframe that contains your node ids under "node_ids" and role assingments under "majority_role".
    k : int
        number of groups.

    Returns
    -------
    node_closeness: dict
        dictionary of nodes and their closeness. 

    """
    dataframe = dataframe.set_index("node_id")
    for edge in graph.edges(): 
        u, v = edge 
        nx.set_edge_attributes(graph, {(u,v): {'distance': 1 - graph[u][v]["weight"]}})
    node_closeness = nx.closeness_centrality(graph, distance='distance')
    group_closeness = {}
    group_closeness_std = {}
    for n in range(k):
        scores = []
        for node in node_closeness.keys():
            if dataframe.at[node, "majority_role"] == n:
                scores.append(node_closeness[node])
        avg = sum(scores) / len(scores) if len(scores) > 0 else print(f"Error! group {n} is showing a score list with length <= 0")
        group_closeness[n] = avg
        group_closeness_std[n] = np.std(scores)
    return node_closeness, group_closeness, group_closeness_std

weighted_closeness, weighted_group_closeness, weighted_group_closeness_std = group_closeness(graph_weighted, weighted_df, 6)
print(weighted_group_closeness)    

def group_pagerank(graph, dataframe, k): 
    """

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        
    dataframe : pandas.DataFrame
        A dataframe that contains your node ids under "node_ids" and role assingments under "majority_role".
    k : int
        number of groups.

    Returns
    -------
    .

    """
    dataframe = dataframe.set_index("node_id")
    node_pagerank = nx.pagerank(graph, weight= 'weight')
    group_pagerank = {}
    group_pagerank_std = {}
    for n in range(k):
        scores = []
        for node in node_pagerank.keys():
            if dataframe.at[node, "majority_role"] == n:
                scores.append(node_pagerank[node])
        avg = sum(scores) / len(scores) if len(scores) > 0 else print(f"Error! group {n} is showing list with length <= 0")
        group_pagerank[n] = avg
        group_pagerank_std[n] = np.std(scores)
    return node_pagerank, group_pagerank, group_pagerank_std

weighted_pagerank, weighted_group_pagerank, weighted_group_pagerank_std = group_pagerank(graph_weighted, weighted_df, 6)

print(weighted_group_pagerank)
print(weighted_group_pagerank_std)

def group_indegree(graph, dataframe, k):
    """

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        
    dataframe : pandas.DataFrame
        A dataframe that contains your node ids under "node_ids" and role assingments under "majority_role".
    k : int
        number of groups.

    Returns
    -------
    .

    """
    dataframe = dataframe.set_index("node_id")
    node_indegree = dict(graph.in_degree(graph, weight='weight'))
    group_indegree = {}
    group_indegree_std = {}
    for n in range(k):
        scores = []
        for node in node_indegree.keys():
            if dataframe.at[node, "majority_role"] == n:
                scores.append(node_indegree[node])
        avg = sum(scores) / len(scores) if len(scores) > 0 else print(f"Error! group {n} is showing list with length <= 0")
        group_indegree[n] = avg
        group_indegree_std[n] = np.std(scores)
    return node_indegree, group_indegree, group_indegree_std

weighted_indegree, weighted_group_indegree, weighted_group_indegree_std = group_indegree(graph_weighted, weighted_df, 6)
print(weighted_group_indegree)    
print(weighted_group_indegree_std)    

def group_outdegree(graph, dataframe, k):
    """

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        
    dataframe : pandas.DataFrame
        A dataframe that contains your node ids under "node_ids" and role assingments under "majority_role".
    k : int
        number of groups.

    Returns
    -------
    .

    """
    dataframe = dataframe.set_index("node_id")
    node_outdegree = dict(graph.out_degree(graph, weight='weight'))
    group_outdegree = {}
    group_outdegree_std = {}
    for n in range(k):
        scores = []
        for node in node_outdegree.keys():
            if dataframe.at[node, "majority_role"] == n:
                scores.append(node_outdegree[node])
        avg = sum(scores) / len(scores) if len(scores) > 0 else print(f"Error! group {n} is showing list with length <= 0")
        group_outdegree[n] = avg
        group_outdegree_std[n] = np.std(scores)
    return node_outdegree, group_outdegree, group_outdegree_std

weighted_outdegree, weighted_group_outdegree, weighted_group_outdegree_std = group_outdegree(graph_weighted, weighted_df, 6)
print(weighted_group_outdegree)    
print(weighted_group_outdegree_std) 

#%% Calculating Avergage Year for each EP group
group_numbers = {}

for n in range(6):
    total = 0
    for i in weighted_df.index:
        if weighted_df.at[i, "majority_role"] == n: 
            total += 1
    group_numbers[n] = total

print(group_numbers)

df = weighted_df.set_index("node_id")

group_years = {}


for n in range(6):
    total = []
    for i in df.index: 
        if df.at[i, "majority_role"] == n: 
            year = TM_data.at[i, "firstpub"]
            total.append(year)
    avg = sum(total) / len(total)
    group_years[n] = avg
    
print(group_years)

pub_years = [i for i in TM_data["firstpub"]]
print(np.mean(pub_years))
print(np.std(pub_years))


#%% 
"""
Dispplaying calculated centrality metrics in table.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_group_centrality_table(
    k,
    group_ec_mean, group_ec_std,
    group_bet_mean, group_bet_std,
    group_close_mean, group_close_std,
    group_pr_mean, group_pr_std,
    group_in_mean=None, group_in_std=None,
    group_out_mean=None, group_out_std=None,
    group_labels=None,
    title="Group Centrality Summary",
    fmt="{:.4g}",
    save_png_path=None,
    save_csv_path=None,
    dpi=300,
    figsize=(10, 0.6)
):
    """
    Create a publication-quality table of group-level centrality statistics.
    
    Parameters
    ----------
    k : int
        Number of groups.
    group_*_mean/std : dict[int -> float]
        Per-group means and 'average deviation' (here: std) for each metric.
    group_in_mean/std : dict[int -> float] or None
        Optional in-degree means/stds if you computed them (directed graphs).
    group_labels : list[str] or None
        Optional display labels per group (length k). Defaults to 'Group 0..k-1'.
    title : str
        Figure title.
    fmt : str
        Format string for numbers (applied to mean and std).
    save_png_path : str or None
        If provided, saves the figure as a high-DPI PNG.
    save_csv_path : str or None
        If provided, writes the underlying table to CSV.
    dpi : int
        Figure DPI for saving.
    figsize : tuple(float, float)
        Base width and per-row height (height is multiplied by k internally).
    """
    # Build a tidy DataFrame: one row per group, columns: metric Mean, metric ± (std)
    rows = []
    index = []
    for g in range(k):
        index.append(group_labels[g] if (group_labels and g < len(group_labels)) else f"Group {g}")
        row = {
            "Eigenvector (mean)": group_ec_mean.get(g, float("nan")),
            "Eigenvector (±std)": group_ec_std.get(g, float("nan")),
            "Betweenness (mean)": group_bet_mean.get(g, float("nan")),
            "Betweenness (±std)": group_bet_std.get(g, float("nan")),
            "Closeness (mean)": group_close_mean.get(g, float("nan")),
            "Closeness (±std)": group_close_std.get(g, float("nan")),
            "PageRank (mean)": group_pr_mean.get(g, float("nan")),
            "PageRank (±std)": group_pr_std.get(g, float("nan")),
        }
        if group_in_mean is not None and group_in_std is not None:
            row["In-degree (mean)"] = group_in_mean.get(g, float("nan"))
            row["In-degree (±std)"] = group_in_std.get(g, float("nan"))
        if group_out_mean is not None and group_out_std is not None:                 # ← add
            row["Out-degree (mean)"] = group_out_mean.get(g, float("nan"))
            row["Out-degree (±std)"] = group_out_std.get(g, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows, index=index)

    # String-format mean and std into a single "mean ± std" display column for compactness
    def _combine(m, s):
        try:
            return (fmt.format(m) + " ± " + fmt.format(s)) if pd.notna(m) and pd.notna(s) else ""
        except Exception:
            return ""

    # Build a compact display version
    display_cols = []
    compact = pd.DataFrame(index=df.index)
    for metric in ["Eigenvector", "Betweenness", "Closeness", "PageRank"]:
        mcol = f"{metric} (mean)"
        scol = f"{metric} (±std)"
        if mcol in df and scol in df:
            compact[metric] = [_combine(m, s) for m, s in zip(df[mcol], df[scol])]
            display_cols.append(metric)
    if "In-degree (mean)" in df and "In-degree (±std)" in df:
        compact["In-degree"] = [_combine(m, s) for m, s in zip(df["In-degree (mean)"], df["In-degree (±std)"])]
        display_cols.append("In-degree")
    if "Out-degree (mean)" in df and "Out-degree (±std)" in df:                  # ← add
        compact["Out-degree"] = [_combine(m, s) for m, s in zip(df["Out-degree (mean)"], df["Out-degree (±std)"])]
        display_cols.append("Out-degree")
    

    # Save CSV (both raw numeric and compact text forms) if requested
    if save_csv_path:
        out = df.copy()
        for metric in display_cols:
            mean_col = f"{metric} (mean)" if metric != "In-degree" else "In-degree (mean)"
            std_col  = f"{metric} (±std)" if metric != "In-degree" else "In-degree (±std)"
            if mean_col in out and std_col in out:
                out[f"{metric} (mean±std)"] = compact[metric]
        out.to_csv(save_csv_path, index=True)

    # Matplotlib table: publication settings
    rcParams.update({
        "font.size": 11,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.edgecolor": "black",
    })

    # Scale figure height with number of groups for readability
    fig_h = max(figsize[1] * (k + 2), 2.0)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))
    ax.axis("off")

    # Convert compact DataFrame to a table
    cell_text = compact[display_cols].values.tolist()
    col_labels = display_cols
    row_labels = compact.index.tolist()

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    # Styling: larger header, bold, alternating row colors, tight cell padding
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Column widths (tune if you like)
    ncols = len(col_labels)
    col_w = [1.0 / max(ncols, 1)] * ncols
    for i, w in enumerate(col_w):
        table.auto_set_column_width(col=i)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        # Header row is row == 0 in Matplotlib tables
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_edgecolor("black")
            cell.set_linewidth(1.0)
            cell.set_height(0.10)
        else:
            # Alternating row shading for readability
            if row % 2 == 0:
                cell.set_facecolor((0.97, 0.97, 0.97))
            cell.set_edgecolor("black")
            cell.set_linewidth(0.5)
            cell.set_height(0.085)

    # Title
    ax.set_title(title, pad=12, fontsize=12, weight="bold")

    plt.tight_layout()

    if save_png_path:
        plt.savefig(save_png_path, bbox_inches="tight", facecolor="white")
    return fig, ax, compact  # return compact (display) table for further use
 
# Means and stds from your computed dictionaries
fig, ax, compact = plot_group_centrality_table(
    k=6,
    group_ec_mean=weighted_group_ecs,
    group_ec_std=weighted_ec_std,
    group_bet_mean=weighted_group_betweenness,
    group_bet_std=weighted_group_betweenness_std,
    group_close_mean=weighted_group_closeness,
    group_close_std=weighted_group_closeness_std,
    group_pr_mean=weighted_group_pagerank,
    group_pr_std=weighted_group_pagerank_std,
    group_in_mean=weighted_group_indegree,              # include if desired
    group_in_std=weighted_group_indegree_std,           # include if desired
    group_out_mean=weighted_group_outdegree,
    group_out_std=weighted_group_outdegree_std,
    group_labels=[f"EP {i}" for i in range(6)],         # optional display labels
    title="Centrality by Equitable-Partition Group",
    save_png_path="group_centrality_table.png",         # optional
    save_csv_path="group_centrality_table.csv",         # optional
    dpi=300
)
plt.show()

#%%
"""
This retrieves the ten "most characteristic" nodes of each group adn prepares them to be loaded into a CSV table.
"Most characteristic" is defined as the nodes with the highest stochastic assignment to the category.
"""
_np_call_re = re.compile(r'np\.\w+\(\s*([^)]+?)\s*\)')  # strips np.float64(...)# Regex to strip np.float64(...) → inner number
_np_call_re = re.compile(r'np\.float64\(\s*([^)]+)\s*\)')

def parse_role_prob_string(s: str) -> dict[int, float]:
    """Convert a string like '{0: np.float64(0.25), 1: np.float64(0.15)}' to {0:0.25,1:0.15}."""
    if isinstance(s, dict):
        return s  # already parsed
    if not isinstance(s, str):
        raise TypeError(f"Unexpected type: {type(s)}")
    # Remove the np.float64(...) wrappers
    cleaned = _np_call_re.sub(r'\1', s)
    # Now safely parse to dict
    d = ast.literal_eval(cleaned)
    # Normalize
    return {int(k): float(v) for k, v in d.items()}

weighted_df_copy = weighted_df.copy()
weighted_df_copy['role_probs'] = weighted_df_copy['role_probs'].apply(parse_role_prob_string)

def _parse_cell(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = _np_call_re.sub(r'\1', x)
        return ast.literal_eval(s)
    raise TypeError("role_probs must contain dicts or dict-like strings.")

def _coerce_vals_to_float(d):
    out = {}
    for k, v in d.items():
        try:
            out[k] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        except Exception:
            try:
                out[k] = float(str(v))
            except Exception:
                out[k] = None
    return out

def _get_prob(d, cat):
    # Try exact key
    if cat in d:
        return d[cat]
    # Try string/int variants
    if isinstance(cat, int) and str(cat) in d:
        return d[str(cat)]
    if isinstance(cat, str):
        try:
            ic = int(cat)
            if ic in d:
                return d[ic]
        except Exception:
            pass
    return None

def top_nodes_by_role_probs(
    df: pd.DataFrame,
    k: int = 10,
    *,
    categories=None,
    node_col: str | None = None,
    dropna: bool = False,
    return_diagnostics: bool = True
):
    if "role_probs" not in df.columns:
        raise ValueError("Missing 'role_probs' column.")

    # Normalize dicts and values
    rp = []
    for x in df["role_probs"]:
        d = _coerce_vals_to_float(_parse_cell(x))
        rp.append(d)
    df = df.copy()
    df["__rp__"] = rp

    nodes = df[node_col].tolist() if node_col else df.index.tolist()

    # Infer categories if needed (preserve numeric order first)
    if categories is None:
        keys = set().union(*[set(d.keys()) for d in df["__rp__"]])
        nums = [k for k in keys if isinstance(k, (int, float)) or (isinstance(k, str) and k.isdigit())]
        strs = [k for k in keys if k not in nums]
        # Sort numeric by int value, then others lexicographically
        categories = sorted(nums, key=lambda x: int(x) if isinstance(x, str) else int(x)) + sorted(strs)

    topk_dict, rows = {}, []

    # Diagnostics
    coverage = Counter()
    nonzero = Counter()

    for cat in categories:
        pairs = []
        for node, d in zip(nodes, df["__rp__"]):
            p = _get_prob(d, cat)
            if p is None:
                if dropna:
                    continue
                p = 0.0
            else:
                coverage[cat] += 1
                if p != 0.0:
                    nonzero[cat] += 1
            pairs.append((node, float(p)))

        pairs.sort(key=lambda x: (x[1], str(x[0])), reverse=True)
        topk = pairs[:k]
        topk_dict[cat] = topk
        for rank, (node, prob) in enumerate(topk, start=1):
            rows.append({"category": cat, "node": node, "prob": prob, "rank": rank})

    topk_df = pd.DataFrame(rows, columns=["category", "node", "prob", "rank"])

    if return_diagnostics:
        diag = pd.DataFrame({
            "category": categories,
            "rows_with_value": [coverage[c] for c in categories],
            "rows_with_nonzero": [nonzero[c] for c in categories],
        })
        # Helpful flag: categories that look broken (all zeros or no coverage)
        diag["all_zero_or_missing"] = diag["rows_with_nonzero"].fillna(0).eq(0)
        return topk_dict, topk_df, diag

    return topk_dict, topk_df



top10_dict, top10_df, diag = top_nodes_by_role_probs(weighted_df_copy, node_col = "node_id")

#%% Loading the resultis into a CSV

top10_df["novel_title"] = top10_df["node"].map(TM_data["shorttitle"])

top10_df["author"] = top10_df["node"].map(TM_data["author"])

top10_df.to_csv("top10_novels.csv")

#%%
"""
The following applies a square root transform to the data and assesses it for skew. 
Note that even after the transfrom was applied 
"""
from scipy.stats import skew
def sqrt_transform_metrics(
    df: pd.DataFrame,
    metrics: list,
    *,
    suffix: str = "_sqrt"
):
    """
    Apply log1p (natural log of 1+x) to each metric column.
    Handles zeros automatically; negative values will become NaN.
    Returns transformed DataFrame.
    """
    out = df.copy()
    for m in metrics:
        x = pd.to_numeric(out[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[m + suffix] = np.sqrt(x)
    return out

def group_skew_table(
    df: pd.DataFrame,
    group_col: str,
    transformed_metrics: list,
    *,
    skew_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Compute group-wise skewness for transformed metrics.
    Returns a tidy summary table.
    """
    rows = []
    g = df.groupby(group_col, dropna=False)

    for m in transformed_metrics:
        for grp, sub in g:
            s = pd.to_numeric(sub[m], errors="coerce").dropna()
            n = int(s.shape[0])
            if n >= 3:
                sk = float(skew(s, bias=False))
            else:
                sk = np.nan

            rows.append({
                "metric": m,
                "group": grp,
                "n": n,
                "skew": sk,
                "abs_skew": np.nan if pd.isna(sk) else abs(sk),
                "passes_threshold": (abs(sk) <= skew_threshold) if pd.notna(sk) else False
            })

    out = pd.DataFrame(rows)
    return out[["metric", "group", "n", "skew", "abs_skew", "passes_threshold"]]

metrics = ["eigencent", "pagerank", "betweenness", "closeness", "indegree", "outdegree"]

transformed_metrics = sqrt_transform_metrics(weighted_df, metrics)

skew_summary = group_skew_table(transformed_metrics, group_col="majority_role", 
                                transformed_metrics=[m + "_sqrt" for m in metrics], 
                                skew_threshold=0.8)

#%%
"""
The following performs a Welch's ANOVA on the EP groups over each metric to see if they are distinguished from each other.
"""

# Try pingouin first (Welch ANOVA); fallback to statsmodels if available
_HAS_PINGOUIN = False
try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except Exception:
    _HAS_PINGOUIN = False

def select_numeric_metrics(df: pd.DataFrame, group_col: str, include: list = None):
    """
    Return a clean list of numeric metric columns to analyze.
    - Excludes the group column and obvious non-metrics (e.g., ids, strings, arrays).
    - Optionally restrict to a provided subset via `include`.
    """
    if include is not None:
        # Keep only columns that exist and are numeric
        inc = [c for c in include if c in df.columns]
        num_inc = [c for c in inc if pd.api.types.is_numeric_dtype(df[c])]
        return num_inc

    numeric_cols = [c for c in df.columns
                    if c != group_col and pd.api.types.is_numeric_dtype(df[c])]
    # Heuristic: drop columns that look like identifiers or probabilities vectors
    likely_nonmetrics = {'node', 'node_id', 'id', 'role_probs', 'roles', 'label'}
    numeric_cols = [c for c in numeric_cols if c.lower() not in likely_nonmetrics]
    return numeric_cols

def _prepare_groups(df: pd.DataFrame, group_col: str, metric: str):
    """
    Coerce to numeric, drop NaN/inf, and return list of float64 arrays (one per group),
    keeping only groups with n>=2.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found.")

    sub = df.loc[:, [group_col, metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub.replace([np.inf, -np.inf], np.nan, inplace=True)
    sub.dropna(subset=[group_col, metric], inplace=True)

    groups = []
    sizes = []
    for _, g in sub.groupby(group_col, dropna=False):
        arr = g[metric].to_numpy(dtype=np.float64, copy=False)
        if arr.size >= 2:
            groups.append(arr)
            sizes.append(arr.size)

    if len(groups) < 2:
        raise ValueError(
            f"Not enough valid groups for '{metric}'. "
            f"Need ≥2 groups with ≥2 numeric observations each."
        )
    return groups, sizes

def welch_anova_by_metric(df: pd.DataFrame, group_col: str, metrics: list):
    """
    Welch's ANOVA per metric with robust cleaning and version-aware backends.
    Prefers pingouin; otherwise tries statsmodels if available.
    """
    rows = []

    # Try to import statsmodels backend (only if pingouin not available)
    _use_statsmodels = False
    if not _HAS_PINGOUIN:
        try:
            from statsmodels.stats.oneway import anova_oneway  # newer statsmodels
            _use_statsmodels = True
        except Exception:
            _use_statsmodels = False

    for m in metrics:
        try:
            if _HAS_PINGOUIN:
                sub = df.loc[:, [group_col, m]].copy()
                sub[m] = pd.to_numeric(sub[m], errors="coerce")
                sub.replace([np.inf, -np.inf], np.nan, inplace=True)
                sub.dropna(subset=[group_col, m], inplace=True)
                if sub[group_col].nunique(dropna=False) < 2:
                    raise ValueError(
                        f"Not enough valid groups for '{m}'. Need ≥2 groups."
                    )
                aov = pg.welch_anova(dv=m, between=group_col, data=sub)
                # pingouin returns one-row DF
                rows.append({
                    "metric": m,
                    "F": float(aov.loc[0, "F"]),
                    "df1": float(aov.loc[0, "ddof1"]),
                    "df2": float(aov.loc[0, "ddof2"]),
                    "p": float(aov.loc[0, "p-unc"]),
                    "k_groups": int(sub[group_col].nunique(dropna=False)),
                    "ns_per_group": sub.groupby(group_col)[m].size().tolist(),
                })
            elif _use_statsmodels:
                groups, sizes = _prepare_groups(df, group_col, m)
                from statsmodels.stats.oneway import anova_oneway
                res = anova_oneway(groups, use_var="unequal", welch_correction=True)
                rows.append({
                    "metric": m,
                    "F": float(res.statistic),
                    "df1": float(res.df_num),
                    "df2": float(res.df_denom),
                    "p": float(res.pvalue),
                    "k_groups": len(groups),
                    "ns_per_group": [int(s) for s in sizes],
                })
            else:
                raise ImportError(
                    "Welch ANOVA backend not available. "
                    "Install `pingouin` (recommended) or upgrade `statsmodels` "
                    "to a version with statsmodels.stats.oneway.anova_oneway."
                )
        except Exception as e:
            rows.append({
                "metric": m,
                "F": np.nan, "df1": np.nan, "df2": np.nan, "p": np.nan,
                "k_groups": np.nan, "ns_per_group": None,
                "error": str(e),
            })

    return pd.DataFrame(rows)


trans_metrics=[m + "_sqrt" for m in metrics]
wanova_df = welch_anova_by_metric(transformed_metrics, 'majority_role', transformed_metrics)

#%%
"""
The following performs as PERMANOVA to try to detect if the groups are differentiated by a profile of metrics.
"""

def run_permanova(df: pd.DataFrame,
                  group_col: str,
                  metrics: list,
                  *,
                  distance: str = "euclidean",
                  n_perm: int = 999):
    from skbio.stats.distance import DistanceMatrix, permanova

    # --- checks & cleaning ---
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"Metric '{m}' not found in DataFrame.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    sub = df.loc[:, [group_col] + metrics].copy()
    for m in metrics:
        sub[m] = pd.to_numeric(sub[m], errors="coerce")
    sub = sub.dropna(subset=[group_col] + metrics)

    # need ≥2 groups with ≥2 obs each
    counts = sub[group_col].value_counts(dropna=False)
    if (counts >= 2).sum() < 2:
        raise ValueError("Need at least 2 groups with ≥2 observations each for PERMANOVA.")

    # fix alignment by resetting the row order and using a vector for grouping
    sub = sub.reset_index(drop=True)
    X = sub[metrics].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    D = squareform(pdist(Xz, metric=distance))

    ids = sub.index.astype(str).tolist()  # any unique strings are fine
    grouping_vec = sub[group_col].astype(str).to_numpy()  # <-- vector, not Series

    dm = DistanceMatrix(D, ids=ids)
    res = permanova(distance_matrix=dm, grouping=grouping_vec, permutations=n_perm)
    return res


perma = run_permanova(transformed_metrics, "majority_role", trans_metrics)


#%%