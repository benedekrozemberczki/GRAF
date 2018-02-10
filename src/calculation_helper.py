import networkx as nx
import numpy as np
import random
import math
import community
from collections import Counter
from tqdm import tqdm
from sklearn.cluster import KMeans

def normalized_overlap(g, node_1, node_2):
    """
    Function to calculate the normalized neighborhood overlap.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2)))) + 1
    unio = len(set(nx.neighbors(g, node_1)).union(set(nx.neighbors(g, node_2))))
    overlap_value = float(inter) / float(unio)
    return overlap_value

def overlap(g, node_1, node_2):
    """
    Function to calculate the neighborhood overlap.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2)))) + 1
    return float(inter)

def unit(g, node_1, node_2):
    """
    Function to calculate the 'unit' weight.
    """
    return 1

def min_norm(g, node_1, node_2):
    """
    Function to calculate the minimum neighborhood overlap.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    min_norm = min(len(set(nx.neighbors(g, node_1))), len(set(nx.neighbors(g, node_2))))
    min_overlap = float(inter) / float(min_norm)
    return min_overlap

def overlap_generator(metric, graph):
    """
    """
    edges = nx.edges(graph)
    edges = edges + map(lambda x: (x[1], x[0]), edges)
    return {edge: metric(graph, edge[0], edge[1]) for edge in tqdm(edges)}

def classical_modularity_calculator(graph, embedding, args):
    """
    """
    kmeans = KMeans(n_clusters=args.cluster_number, random_state=0, n_init=1).fit(embedding)
    assignments = {i: int(kmeans.labels_[i]) for i in range(0, embedding.shape[0])}
    modularity = community.modularity(assignments, graph)
    return modularity, assignments
