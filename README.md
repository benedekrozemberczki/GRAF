GRAF
======================
[![codebeat badge](https://codebeat.co/badges/84fde798-824b-4585-b633-bf9dbc2ca83e)](https://codebeat.co/projects/github-com-benedekrozemberczki-graf-master) [![repo size](https://img.shields.io/github/repo-size/benedekrozemberczki/ClusterGCN.svg)](https://github.com/benedekrozemberczki/ClusterGCN/archive/master.zip) [![benedekrozemberczki](https://img.shields.io/twitter/follow/benrozemberczki?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=benrozemberczki)
<p align="justify">
GRAF is a graph embedding algorithm which learns a clustering based on features extracted with an inner product graph factorization machine. The procedure places nodes in an abstract feature space where the inner product of node features reconstructs the neighbourhood overlap matrix. GRAF is a specific application of an inner product factorization machine. The implementation supports GPU use.
</p>
<p align="center">
  <img width="700" src="plain.jpeg">
</p>


This repository provides a reference implementation for **GRAF** as it is used as a benchmark in the paper:
> **GEMSEC: Graph Embedding with Self Clustering.**
> [Benedek Rozemberczki](http://homepages.inf.ed.ac.uk/s1668259/), [Ryan Davies](https://www.inf.ed.ac.uk/people/students/Ryan_Davies.html), [Rik Sarkar](https://homepages.inf.ed.ac.uk/rsarkar/) and [Charles Sutton](http://homepages.inf.ed.ac.uk/csutton/) .
> arXiv, 2018.
>https://arxiv.org/abs/1802.03997

### Citing

If you find GRAF useful in your research, please consider citing the following paper:

>@misc{rozemberczki2018GEMSEC,    
  author={Benedek Rozemberczki and Ryan Davies and Rik Sarkar and Charles Sutton},    
  title={GEMSEC: Graph Embedding with Self Clustering},   
  year={2018},    
  eprint={arXiv:1802.03997}}

### Requirements

The codebase is implemented in Python 3.5.3. Package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
tensorflow-gpu    1.12.1
jsonschema        2.6.0
texttable         1.5.0
python-louvain    0.11
```

### Datasets
<p align="justify">
The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for the `Facebook Politicians` dataset is included in the  `data/` directory.
</p>

### Logging
<p align="justify">
The models are defined in a way that parameter settings and cluster quality is logged in every single epoch. Specifically we log the followings:
</p>

```
1. Hyperparameter settings.     We save each hyperparameter used in the experiment.
2. Cost per epoch.              Reconstruction cost is stored depending on the model type.
3. Cluster quality.             Measured by modularity. We calculate it based on k-means clustering per epoch.
4. Runtime.                     We measure the time needed for optimization -- measured by seconds.
```

### Options

Learning of the embedding is handled by the `src/factorizer.py` script which provides the following command line arguments.

#### Input and output options

```
  --input                STR       Input graph path.                                 Default is `data/politician_edges.csv`.
  --embedding-output     STR       Embeddings path.                                  Default is `output/embeddings/politician_embedding.csv`.
  --cluster-mean-output  STR       Cluster centers path.                             Default is `output/cluster_means/politician_means.csv`.
  --log-output           STR       Log path.                                         Default is `output/logs/politician.log`.
  --assignment-output    STR       Node-cluster assignment dictionary path.          Default is `output/assignments/politician.json`.
  --dump-matrices        BOOL      Whether the trained model should be saved.        Default is `True`.
  --model                STR       Model used.                                       Default is `GRAF`.
```

#### Model options

```
  --epochs                      INT       Number of epochs.                       Default is 10.
  --batch-size                  INT       Number of edges in batch.               Default is 128.
  --target-weighting            STR       Target edge weight strategy.            Default is `overlap`.
  --regularization-weighting    STR       Regularization weighing strategy.       Default is `normalized_overlap`.
  --dimensions                  INT       Number of dimensions.                   Default is 16.
  --initial-learning-rate       FLOAT     Initial learning rate.                  Default is 0.01.
  --minimal-learning-rate       FLOAT     Final learning rate.                    Default is 0.001.
  --annealing-factor            FLOAT     Annealing factor for learning rate.     Default is 1.0.
  --lambd                       FLOAT     Weight regularization penalty.          Default is 2**-4.
  --cluster-number              INT       Number of clusters.                     Default is 20.
  --initial-gamma               FLOAT     Initial clustering cost weight.         Default is 0.1.
  --regularization-noise        FLOAT     Gradient noise.                         Default is 10**-8.
```

### Examples
<p align="justify">
The following commands learn a graph embedding and cluster center and writes them to disk. The node representations are ordered by the ID.
</p>
<p align="justify">
Creating a GRAF embedding of the default dataset with the default hyperparameter settings. Saving the embedding, cluster centres and the log file at the default path.
</p>

```
python src/factorizer.py
```

Turning off the model saving.

```
python src/factorizer.py --dump-matrices False
```

Creating an embedding of an other dataset the `Facebook Companies`. Saving the output and the log in a custom place.

```
python src/factorizer.py --input data/company_edges.csv  --embedding-output output/embeddings/company_embedding.csv --log-output output/cluster_means/company_means.csv
```

Creating a clustered embedding of the default dataset in 128 dimensions and 10 cluster centers.

```
python src/factorizer.py --dimensions 128 --cluster-number 10
```
------------------------------------------------------------------------------

**License**

- [GNU License](https://github.com/benedekrozemberczki/GRAF/blob/master/LICENSE)

------------------------------------------------------------------------------
