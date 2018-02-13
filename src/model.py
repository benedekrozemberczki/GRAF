import random
import numpy as np
import math
import time

import networkx as nx
import tensorflow as tf

from tqdm import tqdm

from calculation_helper import normalized_overlap, overlap, unit, min_norm, overlap_generator
from calculation_helper import classical_modularity_calculator
from print_and_read import json_dumper, log_setup, initiate_dump, tab_printer, epoch_printer, log_updater

class Model(object):
    """
    Abstract model class.
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

class Factor(Model):
    """
    Linear Inner Product Graph Factorization machine.
    """
    def __init__(self, args, graph, **kwargs):
        """
        We need the arguments and the graph object.
        """
        super(Factor, self).__init__(**kwargs)

        self.args = args
        self.graph = graph
        self.weights = overlap_generator(overlap, self.graph)
        self.nodes = self.graph.nodes()
        self.vocab_size = len(self.nodes)
        self.true_step_size = ((len(self.weights.keys()) / 2) * args.batch_size * self.args.epochs)
        self.edges = nx.edges(self.graph)
        self.build()

    def _build(self):
        """
        Method to create the computational graph.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            
            self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
            self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])
    
            self.overlap = tf.placeholder(tf.float32, shape=[None])


            self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                                -0.1/self.args.dimensions, 0.1/self.args.dimensions), name="embed_matrix")

            self.embedding_bias = tf.Variable(tf.random_uniform([self.vocab_size,1],
                                                -0.1/self.args.dimensions, 0.1/self.args.dimensions), name="embed_bias")

            self.embedding_left = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_left, max_norm=1) 
            self.embedding_right = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_right, max_norm=1) 
    
            self.bias = tf.nn.embedding_lookup(self.embedding_bias, self.edge_indices_left, max_norm=1)

            self.embedding_predictions = tf.reduce_sum(tf.multiply(self.embedding_left,self.embedding_right), axis=1) + self.bias
            
            self.main_loss = tf.reduce_mean(tf.square(tf.subtract(self.overlap,self.embedding_predictions)))
            self.regul_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.embedding_right-self.embedding_left), axis=1))
            
            self.loss = self.main_loss + self.args.lambd*self.regul_loss
            
            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step=self.batch)
    
            self.init = tf.global_variables_initializer()

    def feed_dict_generator(self, edges, step):
        
        """
        Method to generate left and right handside matrices, proper time index and overlap vector.
        """

        reverse_edges = {b: random.random() for a, b in edges}
        edges.sort(key=lambda item: reverse_edges[item[1]])
        
        left_nodes = np.array(map(lambda x: x[0], edges))
        right_nodes = np.array(map(lambda x: x[1], edges))
        overlaps = np.array(map(lambda x: self.weights[(x[0], x[1])], edges))

        feed_dict = {self.edge_indices_left: left_nodes,
                     self.edge_indices_right: right_nodes,
                     self.overlap: overlaps,
                     self.step: float(step)}

        return feed_dict

    def train(self):
        """
        Method for training the embedding.
        """
        self.current_step = 0
        self.log = log_setup(self.args)

        with tf.Session(graph=self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.args.epochs):
                
                random.shuffle(self.edges)
                self.optimization_time = 0 
                self.average_loss = 0
                epoch_printer(repetition)
                
                for i in tqdm(range(0,len(self.edges)/self.args.batch_size)):
                    self.current_step = self.current_step + 1
                    feed_dict = self.feed_dict_generator(self.edges[i*self.args.batch_size:(i+1)*self.args.batch_size], self.current_step)
                    start = time.time()
                    _, loss = session.run([self.train_op , self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                self.average_loss = self.average_loss/self.vocab_size
                self.embedding_out = self.embedding_matrix.eval()
                self.modularity_score, assignments = classical_modularity_calculator(self.graph, self.embedding_out, self.args)
                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time, self.modularity_score)
                tab_printer(self.log)

        initiate_dump(self.log, assignments, self.args, self.embedding_out)
