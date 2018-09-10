from collections import defaultdict

import numpy

from scipy.io import loadmat
from neo4j.v1 import GraphDatabase
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k, v in iteritems(G)}


embeddings_file = "example_graphs/blogcatalog.embeddings"
matfile = "example_graphs/blogcatalog.mat"

# 1. Load Embeddings
model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

# 2. Load labels
mat = loadmat(matfile)
A = mat["network"]
graph = sparse2graph(A)
labels_matrix = mat["group"]
labels_count = labels_matrix.shape[1]

# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)

features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])
print(features_matrix)

add_labels_query = """
UNWIND $pairs AS pair
WITH pair.node AS node, pair.embedding AS embedding

MATCH (n:Node {id: node})
SET n.referenceEmbedding = embedding
"""

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

params = []
with driver.session() as session:
    for node in range(len(graph)):
        params.append({"node": long(node), "embedding": model[str(node)].tolist()})

        if len(params) == 1000:
            session.run(add_labels_query, {"pairs": params})
            params = []
    session.run(add_labels_query, {"pairs": params})