# Script by Carla Silva 2021 :: TSP Quantum implementation based on:
# Hopfield-Tank TSP formulation https://link.springer.com/content/pdf/10.1007/BF00339943.pdf
""" TSP 

    Formulation of the problem for a graph random G=(V,E) with a number of cities n.

"""
#### NOTE: WORK, GIVES OPTIMAL SOLUTIONS UNTIL N<=6.

from pyqubo import Array, solve_qubo, Constraint
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys
import random
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import neal
import pandas as pd
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dimod
import dwave.inspector

def plot_city(cities, n, sol = {}):
    n_city = len(cities)
    cities_dict = dict(cities)

    G = nx.Graph()
    for city in cities_dict:
        G.add_node(city)
                    
    for i in range(n_city):
        for j in range(n_city):
            G.add_edge(i, j,distance=round(dist(i, j, cities),2))

    f = plt.figure()
    pos = nx.spring_layout(G) 
    labels = nx.get_edge_attributes(G,'distance')
    nx.draw(G,pos, with_labels=1, node_size=400, font_weight='bold', font_color='w', node_color = 'black', edge_color = 'black')
    nx.draw_networkx_edge_labels(G,pos,label_pos=0.2, edge_labels=labels,font_color='black')
    plt.axis("off")
    f.savefig('figAllDistancesN'+str(n)+'.pdf', bbox_inches='tight')

    G = nx.Graph()
    for city in cities_dict:
        G.add_node(city)
        
    # draw path
    city_order = []
    for i, v in sol.items():
        if v == 1:
            city_order.append(int(i[2]))
            city_order.append(int(i[5]))
    city_order = list(city_order)
    n_city = len(city_order)            
    for k in range(0,n_city-1,2):
        i = city_order[k]
        j = city_order[k+1]
        G.add_edge(i, j, distance=round(dist(i, j, cities),2))

    sumD = 0
    for u,v in G.edges:
        sumD += G[u][v]['distance']

    f = plt.figure()
    pos = nx.spring_layout(G) 
    labels = nx.get_edge_attributes(G,'distance')
    nx.draw(G,pos, with_labels=1, node_size=400, font_weight='bold', font_color='w', node_color = 'orange', edge_color = 'orange')
    nx.draw_networkx_edge_labels(G,pos,label_pos=0.5, edge_labels=labels,font_color='orange')
    plt.axis("off")
    f.savefig('figN'+str(n)+'Distance='+str(sumD)+'.pdf', bbox_inches='tight')

def dist(i, j, cities):
    pos_i = np.array(cities[i][1])
    pos_j = np.array(cities[j][1])
    return np.linalg.norm(np.subtract(pos_i,pos_j)) # Euclidean distance

def exp1(n, v, A):
    exp = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if (k != j):
                    exp += v[i,j]*v[i,k]

    exp = (A/2) * Constraint(exp, label="exp")
    return(exp)

def exp2(n, v, B):
    exp = 0.0
    for j in range(n):
        for i in range(n):
            for k in range(n):
                if (k != i):
                    exp += v[i,j]*v[k,j]

    exp = (B/2) * Constraint(exp, label="exp")
    return(exp)

def exp3(n, v, C):
    exp = 0.0
    for i in range(n):
        for j in range(n):
                exp += (v[i,j]-n)**2

    exp = (C/2) * Constraint(exp, label="exp")
    return(exp)

def exp4(n, v, D):
    exp = 0.0
    for i in range(n):
        for j in range(n):
            if (j != i):
                for k in range(n-1):
                    exp += dist(i,j,cities)*(v[i,k])*(v[j,k+1]+v[j,k-1])

    exp = (D/2) * Constraint(exp, label="exp")
    return(exp)

if __name__ == "__main__":


    """## Traveling Salesman Problem (TSP)

    Find the shortest route that visits each city and returns to the origin city.
    """

    n = int(sys.argv[1]) # Nodes/Cities

    """Prepare binary vector with  bit $(i, j)$ representing to visit $j$ city at time $i$"""

    v = Array.create('v', (n, n), 'BINARY')

    
    random.seed(123)
    a = tuple((random.randint(0,50),random.randint(0,50)) for i in range(n))
    b = tuple((i,a[i]) for i in range(n))
    cities = list(b)
    
    # maximum distance
    maxdist = 0
    for i in range(n):
        for j in range(n):
            if dist(i,j,cities) > maxdist:
                maxdist = dist(i,j,cities)

    n_city = len(cities)
    cities_dict = dict(cities)

    G = nx.Graph()
    for city in cities_dict:
        G.add_node(city)
                    
    for i in range(n_city):
        for j in range(n_city):
            G.add_edge(i, j,distance=round(dist(i, j, cities),2))

    dU = G.size(weight="weight")
    dL = dU/2
    C = 1
    D = 1 / dU
    B = (3 * dU) + C
    A = B - (D * dL)
    print("dU,dL,A,B,C,D",dU,dL,A,B,C,D)
    orig_stdout = sys.stdout

    f = open('tspResults_'+'n'+str(n)+'.txt', 'w')
    sys.stdout = f

    start_time = time.time()

    print("--------------------------------------------------------------------")
    print("\n# TSP PROBLEM WITH n CITIES ON QUANTUM SOLVER #\n")
    print("--------------------------------------------------------------------")

    print("--------------------------------------------------------------------")
    print("1st expression:")
    print("--------------------------------------------------------------------")
    print(exp1(n, v, A))

    print("--------------------------------------------------------------------")
    print("2nd expression:")
    print("--------------------------------------------------------------------")
    print(exp2(n, v, B))

    print("--------------------------------------------------------------------")
    print("3rd expression:")
    print("--------------------------------------------------------------------")
    print(exp3(n, v, C))

    print("--------------------------------------------------------------------")
    print("4th expression:")
    print("--------------------------------------------------------------------")
    print(exp4(n, v, D))

    # Define hamiltonian H
    H = exp1(n, v, A) + exp2(n, v, B)  + exp3(n, v, C)  + exp4(n, v, D)


    # Compile model
    model = H.compile()

    # Create QUBO
    qubo, offset = model.to_qubo()

    print("--------------------------------------------------------------------")
    print("\nQUBO:\n")
    print("--------------------------------------------------------------------")

    print(qubo)

    print("--------------------------------------------------------------------")
    print("\nD-WAVE OUTPUT:\n")
    print("--------------------------------------------------------------------")

    sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi', token='DEV-dcbd959c9c01feecb7be82680371fc4dce2b1b9e', solver={'topology__type': 'pegasus'}))


    # Submit to the D-Wave with nr number of reads
    # Reads number
    nr = 10000
    # Chain strength
    c = max(qubo.values()) + 10 
 
    print("chain break: ",c)

    start_time = time.time()

    response = sampler.sample_qubo(qubo, num_reads = nr, auto_scale=True, return_embedding=True, chain_strength = c, answer_mode = "raw")        

    print("QPU access time (us):\t", response.info['timing']['qpu_access_time'])

    elapsed_time = time.time() - start_time

    print("Wall time (us):\t\t", elapsed_time*1000000)

    # Inspect
    dwave.inspector.show(response)

    # create dataframe if we want to store all values
    df = []
    minD = sys.maxsize # long value
    for datum in response.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):
        df.append({"Sample": datum.sample, "Energy": datum.energy, "Occurrences": datum.num_occurrences, "Chain_break_fractions": datum.chain_break_fraction})
        print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences,"Chain break fractions:", datum.chain_break_fraction)

        if (sum(datum.sample.values())==n): # Consider only instances with n nodes (same as the n of the problem). 
            s = 0
            for i, v in datum.sample.items():
                if ((int(i[2]) != int(i[5])) and (int(v) == 1)): # Found an edge. Check if i != j (different nodes).
                    s = s + 1
            if ((s == n)): # Check the number of edges = number of nodes.
                ### Draw graph
                n_city = len(cities)
                cities_dict = dict(cities)
                G = nx.Graph()
                for city in cities_dict:
                    G.add_node(city)        
                # draw path
                city_order = []
                for i, v in datum.sample.items():
                    if v == 1:
                        city_order.append(int(i[2]))
                        city_order.append(int(i[5]))

                city_order = list(city_order)
                n_city = len(city_order)            
                for k in range(0,n_city-1,2):
                    i = city_order[k]
                    j = city_order[k+1]
                    G.add_edge(i, j, distance=round(dist(i, j, cities),2))

                sumD = 0
                for u,v in G.edges:
                    sumD += G[u][v]['distance']

                # Check if graph is connected and with a cycle.
                try:
                    G1 = list(nx.cycle_basis(G.to_undirected())) # check if is one cicle with n nodes.
                    if (nx.is_connected(G) and nx.find_cycle(G) and (len(G1[0])==n) and (sumD < minD)):
                        minD = sumD
                        minE = datum.energy
                        maxO = datum.num_occurrences
                        sample = datum.sample
                        chain = datum.chain_break_fraction
                except:
                    break
        
    df = pd.DataFrame(df)
    df.to_csv('TSP'+str(n)+'.csv',index=False)
    pd.set_option('display.float_format', lambda x: '%.20f' % x)
    pd.options.display.max_colwidth = 10000
    print(df.to_string(index=False))

    print("--------------------------------------------------------------------")
    print("\nSAMPLE WITH MINIMUM ENERGY AND MAXIMUM OCCURRENCES:\n")
    print("--------------------------------------------------------------------")
    print(sample, "Energy: ", minE, "Occurrences: ", maxO, "Chain break fractions:", chain)

    plot_city(cities, n, sample)

    print("--------------------------------------------------------------------")
    print("\nTIME\n")
    print("--------------------------------------------------------------------")

    print("Times:\t\t", response.info)
    
    sys.stdout = orig_stdout
    

