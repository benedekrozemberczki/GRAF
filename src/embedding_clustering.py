from parser import parameter_parser
from print_and_read import graph_reader
from model import GRAFCODEWithRegularization, GRAFCODE, GRAFWithRegularization, GRAF

def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    """
    graph = graph_reader(args.input)
    if args.model == "GRAFCODEWithRegularization":
        model = GRAFCODEWithRegularization(args, graph)
    elif args.model == "GRAFCODE":
        model = GRAFCODE(args, graph)        
    elif args.model == "GRAFWithRegularization":
        model = GRAFWithRegularization(args, graph)
    else:
        model = GRAF(args, graph)   
    model.train()

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
