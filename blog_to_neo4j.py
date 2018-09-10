from scipy.io import loadmat
from neo4j.v1 import GraphDatabase


def _csr_gen_triples(A):
    """Converts a SciPy sparse matrix in **Compressed Sparse Row** format to
    an iterable of weighted edge triples.

    """
    nrows = A.shape[0]
    data, indices, indptr = A.data, A.indices, A.indptr
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield i, indices[j], data[j]


def _csc_gen_triples(A):
    """Converts a SciPy sparse matrix in **Compressed Sparse Column** format to
    an iterable of weighted edge triples.

    """
    ncols = A.shape[1]
    data, indices, indptr = A.data, A.indices, A.indptr
    for i in range(ncols):
        for j in range(indptr[i], indptr[i + 1]):
            yield indices[j], i, data[j]


def _coo_gen_triples(A):
    """Converts a SciPy sparse matrix in **Coordinate** format to an iterable
    of weighted edge triples.

    """
    row, col, data = A.row, A.col, A.data
    return zip(row, col, data)


def _dok_gen_triples(A):
    """Converts a SciPy sparse matrix in **Dictionary of Keys** format to an
    iterable of weighted edge triples.

    """
    for (r, c), v in A.items():
        yield r, c, v


def _generate_weighted_edges(A):
    """Returns an iterable over (u, v, w) triples, where u and v are adjacent
    vertices and w is the weight of the edge joining u and v.

    `A` is a SciPy sparse matrix (in any format).

    """
    if A.format == 'csr':
        return _csr_gen_triples(A)
    if A.format == 'csc':
        return _csc_gen_triples(A)
    if A.format == 'dok':
        return _dok_gen_triples(A)
    # If A is in any other format (including COO), convert it to COO format.
    return _coo_gen_triples(A.tocoo())


driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

matfile = "example_graphs/blogcatalog.mat"
mat = loadmat(matfile)

A = mat["network"]
n, m = A.shape

constraint_query = """
CREATE CONSTRAINT ON (n:Node)
ASSERT n.id IS UNIQUE
"""

create_nodes_query = """
UNWIND range(0, $n-1) AS id
MERGE (:Node {id: id})
"""

with driver.session() as session:
    session.run(constraint_query)
    session.run(create_nodes_query, {"n": n})

triples = _generate_weighted_edges(A)

create_relationships_query = """
UNWIND $triples AS triple
WITH triple.node1 AS node1, triple.node2 AS node2, triple.weight AS weight

MATCH (n1:Node {id: node1})
MATCH (n2:Node {id: node2})
MERGE (n1)-[connected:CONNECTED]-(n2)
SET connected.weight = weight
"""

params = []
with driver.session() as session:
    for node1, node2, weight in triples:
        params.append({"node1": long(node1), "node2": long(node2), "weight": long(weight)})

        if len(params) == 1000:
            session.run(create_relationships_query, {"triples": params})
            params = []
    session.run(create_relationships_query, {"triples": params})


label_pairs = _generate_weighted_edges(mat["group"])


add_labels_query = """
UNWIND $triples AS triple
WITH triple.node AS node, triple.label AS label

MATCH (n:Node {id: node})
MERGE (l:Label {id: label})
MERGE (n)-[:LABEL]-(l)
"""

params = []
with driver.session() as session:
    for node, label, _ in label_pairs:
        params.append({"node": long(node), "label": long(label)})

        if len(params) == 1000:
            session.run(add_labels_query, {"triples": params})
            params = []
    session.run(add_labels_query, {"triples": params})