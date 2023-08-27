import sys
import networkx as nx
from networkx.algorithms import connectivity


# Load the file and parse it
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

# Initialize directed graph
G = nx.DiGraph()

# Populate the graph based on the adjacency list
for line in lines:
    parts = line.strip().split(" -> ")
    node = parts[0]
    neighbors = parts[1].split(" ")
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Check if the graph is strongly connected
is_strongly_connected = nx.is_strongly_connected(G)

print(is_strongly_connected)

