from parser import parameter_parser
from print_and_read import graph_reader
from calculation_helper import overlap, overlap_generator, classical_modularity_calculator, normalized_overlap
from model import Factor
import numpy as np
from sklearn.preprocessing import scale
def create_and_run_model(args):
    
    graph = graph_reader(args.input)
    model = Factor(args, graph)
    model.train()

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)



