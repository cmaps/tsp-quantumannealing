# Script by Carla Silva 2020 :: TSP Quantum Version

""" TSP
    Formulation of the problem for a graph G=(V,E) with a number of cities n.
"""

import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = plt.figure()
    # Pegasus graph with size parameter 16 
    G=dnx.pegasus_graph(4)  
    dnx.draw_pegasus(G, node_size=30, node_color='b', node_shape='o', style='-', edge_color='gray', width=0.5, crosses=False)
    f.savefig('pegasus.pdf', bbox_inches='tight')
    f = plt.figure()
    # Pegasus graph with size parameter 2 :: K4,4 subgraphs configuration
    G=dnx.pegasus_graph(2)  
    #G = dnx.pegasus_graph(2, nice_coordinates=True)
    dnx.draw_pegasus(G, node_size=100, node_color='b', node_shape='o', style='-', edge_color='gray', width=0.5, crosses=True)
    f.savefig('pegasus1.pdf', bbox_inches='tight')
    f = plt.figure()
    # Pegasus graph with size parameter 2 :: L configuration
    G=dnx.pegasus_graph(2)  
    dnx.draw_pegasus(G, node_size=100, node_color='b', node_shape='o', style='-', edge_color='gray', width=0.5, crosses=False)
    f.savefig('pegasus2.pdf', bbox_inches='tight')
