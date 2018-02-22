import random
import numpy as np
import math
import time

import networkx as nx
import tensorflow as tf

from tqdm import tqdm

from layers import Factorization, Clustering, Regularization

from calculation_helper import gamma_incrementer
from calculation_helper import overlap_generator
from calculation_helper import neural_modularity_calculator, classical_modularity_calculator
from print_and_read import json_dumper, log_setup, initiate_dump_graf, initiate_dump_grafcode, tab_printer, epoch_printer, log_updater

class Model(object):
    """
    Abstract model class.
    """
    def __init__(self,args,graph):
        """
        Every model needs the same initialization -- args, graph.
        We delete the sampler object to save memory.
        We also build the computation graph up. 
        """
        self.args = args
        self.graph = graph
        self.targets = overlap_generator(self.args.target_weighting, self.graph)
        self.weights = overlap_generator(self.args.regularization_weighting, self.graph)
        self.nodes = self.graph.nodes()
        self.vocab_size = len(self.nodes)
        self.true_step_size = ((len(self.weights.keys()) / 2) * args.batch_size * self.args.epochs)
        self.edges = nx.edges(self.graph)
        self.build()


    def build(self):
        """
        Building the model.
        """
        pass

    def feed_dict_generator(self):
        """
        Creating the feed generator
        """
        pass

    def train(self):
        """
        Training the model.
        """
        pass
        
class GRAFCODEWithRegularization(Model):
    """
    Regularized GRAFCODE class.
    """
    def build(self):
        """
        Method to create the computational graph.
        """        
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.factorization_layer =  Factorization(self.args, self.vocab_size)
            self.cluster_layer = Clustering(self.args)
            self.regularizer_layer = Regularization(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.factorization_layer()+self.gamma*self.cluster_layer(self.factorization_layer)+self.regularizer_layer(self.factorization_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.args, self.graph)

    def feed_dict_generator(self, edges, step, gamma):
        
        """
        Method to generate left and right handside matrices, proper time index and overlap vector.
        """

        left_nodes = np.array(map(lambda x: x[0], edges))
        right_nodes = np.array(map(lambda x: x[1], edges))

        targets = np.array(map(lambda x: self.targets[(x[0], x[1])], edges))
        regularization_weight = np.array(map(lambda x: self.targets[(x[0], x[1])], edges))

        feed_dict = {self.factorization_layer.edge_indices_left: left_nodes,
                     self.factorization_layer.edge_indices_right: right_nodes,
                     self.factorization_layer.target: targets,
                     self.regularizer_layer.regularization_weight: regularization_weight,
                     self.step: float(step),
                     self.gamma: gamma}

        return feed_dict

    def train(self):
        """
        Method for training the embedding, logging and this method is inherited by GEMSEC and DeepWalk variants without an override.
        """ 
        self.current_step = 0
        self.log = log_setup(self.args)
        self.current_gamma = self.args.initial_gamma
        with tf.Session(graph = self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.args.epochs):

                random.shuffle(self.nodes)
                self.optimization_time = 0 
                self.average_loss = 0

                epoch_printer(repetition)
                for i in tqdm(range(0,len(self.edges)/self.args.batch_size)):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma, self.current_gamma, self.true_step_size)
                    feed_dict = self.feed_dict_generator(self.edges[i*self.args.batch_size:(i+1)*self.args.batch_size], self.current_step, self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op , self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss/self.vocab_size
                self.final_embeddings = self.factorization_layer.embedding_matrix.eval()
                if "CODE" in self.args.model: 
                    self.c_means = self.cluster_layer.cluster_means.eval()
                    self.modularity_score, assignments = neural_modularity_calculator(self.graph, self.final_embeddings, self.c_means)
                else:
                    self.modularity_score, assignments = classical_modularity_calculator(self.graph, self.final_embeddings, self.args)
                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time, self.modularity_score)
                tab_printer(self.log)
        if "CODE" in self.args.model: 
            initiate_dump_grafcode(self.log, assignments, self.args, self.final_embeddings, self.c_means)
        else:
            initiate_dump_graf(self.log, assignments, self.args, self.final_embeddings)


class GRAFCODE(GRAFCODEWithRegularization):
    """
    GRAFCODE class.
    """
    def build(self):
        """
        Method to create the computational graph.
        """        
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.factorization_layer =  Factorization(self.args, self.vocab_size)
            self.cluster_layer = Clustering(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.factorization_layer()+self.gamma*self.cluster_layer(self.factorization_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.args, self.graph)

    def feed_dict_generator(self, edges, step, gamma):
        
        """
        Method to generate left and right handside matrices, proper time index and overlap vector.
        """

        left_nodes = np.array(map(lambda x: x[0], edges))
        right_nodes = np.array(map(lambda x: x[1], edges))

        targets = np.array(map(lambda x: self.targets[(x[0], x[1])], edges))

        feed_dict = {self.factorization_layer.edge_indices_left: left_nodes,
                     self.factorization_layer.edge_indices_right: right_nodes,
                     self.factorization_layer.target: targets,
                     self.step: float(step),
                     self.gamma: gamma}

        return feed_dict

class GRAFWithRegularization(GRAFCODEWithRegularization):
    """
    Regularized GRAF class.
    """
    def build(self):
        """
        Method to create the computational graph.
        """        
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.factorization_layer =  Factorization(self.args, self.vocab_size)
            self.regularizer_layer = Regularization(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.factorization_layer()+self.regularizer_layer(self.factorization_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.args, self.graph)

    def feed_dict_generator(self, edges, step, gamma):
        
        """
        Method to generate left and right handside matrices, proper time index and overlap vector.
        """

        left_nodes = np.array(map(lambda x: x[0], edges))
        right_nodes = np.array(map(lambda x: x[1], edges))

        targets = np.array(map(lambda x: self.targets[(x[0], x[1])], edges))
        regularization_weight = np.array(map(lambda x: self.targets[(x[0], x[1])], edges))

        feed_dict = {self.factorization_layer.edge_indices_left: left_nodes,
                     self.factorization_layer.edge_indices_right: right_nodes,
                     self.factorization_layer.target: targets,
                     self.regularizer_layer.regularization_weight: regularization_weight,
                     self.step: float(step),
                     self.gamma: gamma}

        return feed_dict

class GRAF(GRAFWithRegularization):
    """
    GRAF class.
    """
    def build(self):
        """
        Method to create the computational graph.
        """        
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.factorization_layer =  Factorization(self.args, self.vocab_size)
            self.regularizer_layer = Regularization(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.factorization_layer()

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.args, self.graph)

    def feed_dict_generator(self, edges, step, gamma):
        """
        Method to generate left and right handside matrices, proper time index and overlap vector.
        """

        left_nodes = np.array(map(lambda x: x[0], edges))
        right_nodes = np.array(map(lambda x: x[1], edges))

        targets = np.array(map(lambda x: self.targets[(x[0], x[1])], edges))
        regularization_weight = np.array(map(lambda x: self.targets[(x[0], x[1])], edges))

        feed_dict = {self.factorization_layer.edge_indices_left: left_nodes,
                     self.factorization_layer.edge_indices_right: right_nodes,
                     self.factorization_layer.target: targets,
                     self.step: float(step),
                     self.gamma: gamma}

        return feed_dict
