# Script by Carla Silva and Inês Dutra 2020 :: TSP Classical Version

""" TSP 

    Formulation of the problem for a graph random G=(V,E) with a number of cities n.

"""

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

def plot_city(cities, n, sol = {}):
    n_city = len(cities)
    cities_dict = dict(cities)

    G = nx.Graph()
    for city in cities_dict:
        G.add_node(city)
                    
    for i in range(n_city):
        for j in range(n_city):
            if i != j:
                G.add_edge(i, j,distance=round(dist(i, j, cities),2))

    f = plt.figure()
    pos = nx.spring_layout(G) 
    labels = nx.get_edge_attributes(G,'distance')
    nx.draw(G,pos, with_labels=1, node_size=400, font_weight='bold', font_color='w', node_color = 'blue', edge_color = 'blue')
    nx.draw_networkx_edge_labels(G,pos,label_pos=0.2, edge_labels=labels,font_color='blue', font_size=16)
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
    nx.draw(G,pos, with_labels=1, node_size=400, font_weight='bold', font_color='w', node_color = 'green', edge_color = 'green')
    nx.draw_networkx_edge_labels(G,pos,label_pos=0.5, edge_labels=labels,font_color='green', font_size=16)
    plt.axis("off")
    f.savefig('figN'+str(n)+'Distance='+str(sumD)+'.pdf', bbox_inches='tight')

def dist(i, j, cities):
    pos_i = np.array(cities[i][1])
    pos_j = np.array(cities[j][1])
    return np.linalg.norm(np.subtract(pos_i,pos_j)) # Euclidean distance

def alpha(n,h,maxdist):
  return((((n**3)-(2*n**2)+n)*maxdist)+h)

def beta(n,alfa,h):
  return((((n**2)+1)*alfa)+h)

def exp1(n, v, A):
    exp = 0.0
    for i in range(n):
        for j in range(n):
            exp += Constraint(1-v[i,j], label="exp")
    exp = A*exp
    return(exp)

def exp2(n, v, B):
    exp = 0.0
    for i in range(n):
        for j in range(n):
            if (i != j):
                for k in range(n): 
                    exp += Constraint(((v[i,k])*(v[j,k])), label="exp")
    exp = B*exp
    return(exp)

def exp3(n, v, B):
    exp = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if (k != j):
                    exp += Constraint((v[i,j])*(v[i,k]), label="exp")
    exp = B*exp
    return(exp)

def exp4(n, v):
    exp = 0.0
    for i in range(n):
        for j in range(n):
            if (i != j):
                for k in range(n-1): 
                    exp += dist(i,j,cities)*Constraint(((v[i,k]*v[j,k+1])), label="exp") 
    return(exp)

if __name__ == "__main__":


    """## Traveling Salesman Problem (TSP)

    Find the shortest route that visits each city and returns to the origin city.
    """

    n = int(sys.argv[1]) # Nodes/Cities
    e = int(sys.argv[2]) # Sweeps (epsilon)
#    A = float(sys.argv[2]) # Alpha 
#    B = float(sys.argv[3]) # Beta

    h = 0.0000005 #small number

    """Prepare binary vector with  bit $(i, j)$ representing to visit $j$ city at time $i$"""

    v = Array.create('v', (n, n), 'BINARY')
    
    random.seed(123)
    a = tuple((random.randint(1,50),random.randint(1,50)) for i in range(n))
    b = tuple((i,a[i]) for i in range(n))
    cities = list(b)

    import tsp
    t = tsp.tsp([list(i) for i in a])
    print("TSP Python package")
    print(t)  # distance, node index list
    
    # maximum distance
    maxdist = 0
    for i in range(n):
        for j in range(n):
            if dist(i,j,cities) > maxdist:
                maxdist = dist(i,j,cities)

    A = alpha(n,h,maxdist)
    B = beta(n,A,h)


    orig_stdout = sys.stdout

    f = open('tspClassicalResults_'+'n'+str(n)+'A'+str(A)+'B'+str(B)+'.txt', 'w')
    sys.stdout = f

    start_time = time.time()

    print("--------------------------------------------------------------------")
    print("\n# TSP PROBLEM WITH n CITIES ON CLASSICAL SOLVER #\n")
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
    print(exp3(n, v, B))

    print("--------------------------------------------------------------------")
    print("4th expression:")
    print("--------------------------------------------------------------------")
    print(exp4(n, v))

    # Define hamiltonian H
    H = exp1(n, v, A) + exp2(n, v, B) + exp3(n, v, B) + exp4(n, v)


    # Compile model
    model = H.compile()

    # Create QUBO
    qubo, offset = model.to_qubo()

    print("--------------------------------------------------------------------")
    print("\nQUBO:\n")
    print("--------------------------------------------------------------------")

    print(qubo)

    start_time = time.time()

    nr = 10000
    c = max(qubo.values())

    max_abs_value = float(max(abs(v) for v in qubo.values()))
    scale_qubo = {k: float(v) / max_abs_value for k, v in qubo.items()}
    sa = neal.SimulatedAnnealingSampler()
    sa_computation = sa.sample_qubo(scale_qubo, num_reads=nr, num_sweeps=e, seed=123, chain_strength=c)

    elapsed_time = time.time() - start_time

    print("chain break: ", c)

    print("sweeps: ", e)

    print("--------------------------------------------------------------------")
    print("\nCLASSICAL RESULTS:\n")
    print("--------------------------------------------------------------------")

    df = []
    best = 0
    count = 0
    minD = sys.maxsize # long value
    for a,b,c in sa_computation.record:
        decoded_solution, broken, energy = model.decode_solution(a, vartype='BINARY')
        if not broken:
            d = 0
        else:
            d = broken["exp"]["penalty"]
        for key in range(0,n):
            decoded_solution['v']['v'+str(key)] = decoded_solution['v'].pop(key)
        for i in range(0,n):
            for j in range(0,n):
                decoded_solution['v'].update( {'v'+str(i)+str(j) : decoded_solution['v']['v'+str(i)][j]} )
            decoded_solution['v'].pop('v'+str(i))
        df.append({"Sample": decoded_solution['v'], "Energy": energy, "Occurrences": c, "Broken chains": d})

        if (sum(decoded_solution['v'].values())==n): # Consider only instances with n nodes (same as the n of the problem). 
            s = 0
            x = 0
            for i, v in decoded_solution['v'].items():
                if ((int(i[1]) != int(i[2])) and (int(v) == 1)): # Found an edge. Check if i != j (different nodes).
                    s = s + 1
                if ((int(i[1]) == int(i[2])) and (int(v) == 1)):
                    x = 1
            if ((s == n) and (x!=1)): # Check the number of edges = number of nodes.
                ### Draw graph
                n_city = len(cities)
                cities_dict = dict(cities)
                G = nx.Graph()
                for city in cities_dict:
                    G.add_node(city)        
                # draw path
                city_order = []
                for i, v in decoded_solution['v'].items():
                    if v == 1:
                        city_order.append(int(i[1]))
                        city_order.append(int(i[2]))

                city_order = list(city_order)
                n_city = len(city_order)            
                for k in range(0,n_city-1,2):
                    i = city_order[k]
                    j = city_order[k+1]
                    G.add_edge(i, j, distance=round(dist(i, j, cities),2))

                sumD = 0
                for u,v in G.edges:
                    sumD += G[u][v]['distance']

                # Check if graph is connected.
                if (nx.is_connected(G) and (sumD < minD) and nx.find_cycle(G)):
                    minD = sumD
                    best = count

        count = count + 1
    
    df = pd.DataFrame(df)
    df.to_csv('TSP'+str(n)+'A'+str(A)+'B'+str(B)+'.csv',index=False)
    pd.set_option('display.float_format', lambda x: '%.20f' % x)
    pd.options.display.max_colwidth = 10000
    print(df.to_string(index=False))

    print("--------------------------------------------------------------------")
    print("\nSAMPLE WITH MINIMUM ENERGY:\n")
    print("--------------------------------------------------------------------")

    #best = np.argmin(sa_computation.record.energy)
    best_solution = list(sa_computation.record.sample[best])

    print(dict(zip(sa_computation.variables, best_solution)))

    decoded_solution, broken, energy = model.decode_solution(best_solution, vartype="BINARY")
    print("number of broken constraint = {}".format(len(broken)))
    print(broken)
    print("energy = {}".format(energy))

    plot_city(cities, n, dict(zip(sa_computation.variables, best_solution)))

    print("--------------------------------------------------------------------")
    print("\nTIME (sec):\n")
    print("--------------------------------------------------------------------")
    print(elapsed_time,"All results",elapsed_time/nr,"One result")

    sys.stdout = orig_stdout