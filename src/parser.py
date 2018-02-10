import argparse

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook politicians network.
    The default hyperparameters give a good quality representation and good cluster centers without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run Factor.")

    parser.add_argument('--input',
                        nargs = '?',
                        default = './data/politician_edges.csv',
	                help = 'Input graph path.')

    parser.add_argument('--embedding-output',
                        nargs = '?',
                        default = './output/embeddings/politician_embedding.csv',
	                help = 'Embeddings path.')

    parser.add_argument('--log-output',
                        nargs = '?',
                        default = './output/logs/politician.json',
	                help = 'Log path.')

    parser.add_argument('--assignment-output',
                        nargs = '?',
                        default = './output/assignments/politician.json',
	                help = 'Log path.')

    parser.add_argument('--dump-matrices',
                        type = bool,
                        default = True,
	                help = 'Save the embeddings to disk or not. Default is true.')

    parser.add_argument('--epochs',
                        type = int,
                        default = 10,
	                help = 'Number of epochs. Default is 10.')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 32,
	                help = 'Number of dimensions. Default is 32.')

    parser.add_argument('--cluster-number',
                        type = int,
                        default = 20,
	                help = 'Number of clusters. Default is 20.')

    parser.add_argument('--batch-size',
                        type = int,
                        default = 256,
	                help = 'Number of edges in batch. Default is 256.')

    parser.add_argument('--initial-learning-rate',
                        type = float,
                        default = 0.001,
	                help = 'Initial learning rate. Default is 0.001.')

    parser.add_argument('--minimal-learning-rate',
                        type = float,
                        default = 0.00001,
	                help = 'Minimal learning rate. Default is 0.00001.')

    parser.add_argument('--lambd',
                        type = float,
                        default = 0.01,
	                help = 'L1 weight regularization. Default is 0.01.')

    parser.add_argument('--annealing-factor',
                        type = float,
                        default = 1,
	                help = 'Annealing factor. Default is 1.0.')

    return parser.parse_args()

