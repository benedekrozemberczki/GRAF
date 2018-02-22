import tensorflow as tf
import math
import numpy as np

class Factorization:
    """
    Factorization layer class.
    """
    def __init__(self, args, vocab_size):
        """
        Initialization of the layer with proper matrices and biases.
        The input variables are also initialized here.
        """
        self.args = args
        self.vocab_size = vocab_size 

        self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
        self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])
    
        self.target = tf.placeholder(tf.float32, shape=[None])

        self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions), name="embed_matrix")

        self.embedding_bias = tf.Variable(tf.random_uniform([self.vocab_size,1],
                                          -0.1/self.args.dimensions, 0.1/self.args.dimensions), name="embed_bias")        

    def __call__(self):
        """
        Calculating the predictive loss.
        """

        self.embedding_left = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_left, max_norm=1) 
        self.embedding_right = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_right, max_norm=1) 
        self.bias = tf.nn.embedding_lookup(self.embedding_bias, self.edge_indices_left, max_norm=1)
        self.embedding_predictions = tf.reduce_sum(tf.multiply(self.embedding_left,self.embedding_right), axis=1) + self.bias
        return tf.reduce_mean(tf.square(tf.subtract(self.target,self.embedding_predictions)))

class Clustering:
    """
    Latent space clustering class.
    """
    def __init__(self, args):
        """
        Initializing the cluster center matrix.
        """
        self.args = args
        self.cluster_means = tf.Variable(tf.random_uniform([self.args.cluster_number, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))
    def __call__(self, Factorizer):
        """
        Calculating the clustering cost.
        """
        self.clustering_differences = tf.expand_dims(tf.concat([Factorizer.embedding_left,Factorizer.embedding_right],0),1) - self.cluster_means
        self.cluster_distances = tf.norm(self.clustering_differences, ord = 2, axis = 2)
        self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis = 1)
        return tf.reduce_mean(self.to_be_averaged)

class Regularization:
    """
    Smoothness regularization class.
    """
    def __init__(self, args):
        """
        Initializing the indexing variables and the weight vector.
        """
        self.args = args
        self.regularization_weight = tf.placeholder(tf.float32, shape=[None])
    def __call__(self, Factorizer):
        """
        Calculating the regularization cost.
        """
        self.left_features = Factorizer.embedding_left
        self.right_features = Factorizer.embedding_right
        self.regularization_differences = self.left_features - self.right_features + np.random.uniform(self.args.regularization_noise,self.args.regularization_noise, (self.args.batch_size, self.args.dimensions))
        self.regularization_distances = tf.norm(self.regularization_differences, ord = 2,axis=1)
        self.regularization_distances = tf.reshape(self.regularization_distances, [-1])
        self.regularization_loss = tf.reduce_mean(tf.multiply(tf.transpose(self.regularization_weight), self.regularization_distances))
        return self.args.lambd*self.regularization_loss
