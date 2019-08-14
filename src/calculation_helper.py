import math
import random
import community
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter
from sklearn.cluster import KMeans

def normalized_overlap(g, node_1, node_2):
    """
    Function to calculate the normalized neighborhood overlap.
    :param g: NetworkX graph.
    :param node_1: Node index 1.
    :param node_2: Node index 2.
    :return : A normalized neighbourhood overlap score.
    """    
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    unio = len(set(nx.neighbors(g, node_1)).union(set(nx.neighbors(g, node_2))))
    return float(inter)/float(unio)

def overlap(g, node_1, node_2):
    """
    Function to calculate the neighborhood overlap.
    :param g: NetworkX graph.
    :param node_1: Node index 1.
    :param node_2: Node index 2.
    :return : Overlap score.
    """    
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2)))) + 1 
    return float(inter)

def unit(g, node_1, node_2):
    """
    Function to calculate the "unit" weight.
    :param g: NetworkX graph.
    :param node_1: Node index 1.
    :param node_2: Node index 2.
    :return : Unit weight.
    """    
    return 1

def min_norm(g, node_1, node_2):
    """
    Function to calculate the minimum normalized neighborhood overlap.
    :param g: NetworkX graph.
    :param node_1: Node index 1.
    :param node_2: Node index 2.
    :return : Min set size.
    """    
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    min_norm = min(len(set(nx.neighbors(g, node_1))), len(set(nx.neighbors(g, node_2))))
    return float(inter)/float(min_norm)

def overlap_generator(overlap_weighting, graph):
    """
    Function to generate weight for all of the edges.
    :param overlap_weighting: Weighting method.
    :param graph: NetworkX graph object.
    """
    if overlap_weighting == "normalized_overlap":
        overlap_weighter = normalized_overlap
    elif overlap_weighting == "overlap":
        overlap_weighter = overlap
    elif overlap_weighting == "min_norm":
        overlap_weighter = min_norm
    else:
        overlap_weighter = unit
    print(" ")
    print("Weight calculation started.")
    print(" ")
    edges = nx.edges(graph)
    weights = {edge: overlap_weighter(graph, edge[0], edge[1]) for edge in tqdm(edges)}
    weights_prime = {(edge[1],edge[0]): value for edge, value in weights.items()}
    weights.update(weights_prime)
    print(" ")
    return weights

def index_generation(weights, a_random_walk):
    """
    Function to generate overlaps and indices.
    """    
    edges = [(a_random_walk[i], a_random_walk[i+1]) for i in range(len(a_random_walk)-1)]
    edge_set_1 = np.array(range(0,len(a_random_walk)-1))
    edge_set_2 = np.array(range(1,len(a_random_walk)))
    overlaps = np.array([weights[edge] for edge in edges]).reshape((-1,1))
    return edge_set_1, edge_set_2, overlaps

def gamma_incrementer(step, gamma_0, current_gamma, num_steps):
    if step >1:
        exponent = (0-np.log10(gamma_0))/float(num_steps)
        current_gamma = current_gamma * (10 **exponent)
    return current_gamma

def neural_modularity_calculator(graph, embedding, means):
    """
    Function to calculate the GEMSEC cluster assignments.
    """    
    assignments = {}
    for node in graph.nodes():
        positions = means-embedding[node,:]
        values = np.sum(np.square(positions),axis=1)
        index = np.argmin(values)
        assignments[int(node)]= int(index)
    modularity = community.modularity(assignments,graph)
    return modularity, assignments

def classical_modularity_calculator(graph, embedding, args):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """    
    kmeans = KMeans(n_clusters=args.cluster_number, random_state=0, n_init = 1).fit(embedding)
    assignments = {int(i): int(kmeans.labels_[i]) for i in range(embedding.shape[0])}
    modularity = community.modularity(assignments,graph)
    return modularity, assignments
