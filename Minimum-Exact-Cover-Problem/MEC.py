"""(See "Ising formulations of many NP problems" by Andrew Lucas)

    - Given n, u, s, return a set U of u random elements, 
      chosen arbitrarily between 0 and n-1, and a dictionary
      containing s subsets of U, whose union is U.

    - Create the graph representing the problem, where an edge between 
      nodes i,j means that the subsets corresponding to i,j have 
      a non-zero intersection.
 
    - Find the minimum energy and its corresponding state(s) 
      iterating over all possible states.
      
      N.B.: If the variable build_H is set to True, 
            the hamiltonian matrix H of the problem is built 
            and energies are computed as expectation values <x|H|x>.

            If the variable build_H is set to False, the energies are 
            straightforwardly computed (faster way).
"""

import itertools
import random
import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


# **********************************************************************

def MEC_instance(u, s, n, print_instance, create_graph):
    """
    Arguments
    ---------
        u : number of U's elements.
        s : number of subsets of U, whose union is U.
        n : the maximum number that can appear in the sets will be n-1.
        print_instance : if True, prints out some information of the instance.
        create_graph : if True, creates a graph to visualize the instance.


    Return
    ------
        U : a set of u elements.
        subsets : a dictionary containing s subsets of U, whose union is U.
    """
    
    possible_numbers_range = range(n)


    # Create U.
    try:
        U = set(random.sample(possible_numbers_range, u)) # Sample without repetitions
    except ValueError as e:
        print(f"\n!!! Explanation: n={n}, u={u}. It must be n >= u", 
                "as we are sampling without replacement !!!\n")
        raise


    # Create a random list of subsets of U.
    L = []
    control = set()
    for i in range(s - 1):
        sub_size = random.randrange(1, u) # select a random element from range(u)
        sub = set(random.sample(U, sub_size))

        # If sub is already in L, repeat the i-th step.
        if any(sub == x for x in L): 
            i = i-1
            print("Re-extracting sub...")
            continue
        
        L += [sub]
        control |= sub # union operation

    
    # Make sure that the union of the subsets is equal to U.
    rest = U - control     
    if rest:
        L += [rest]
    else:
        sub_size = random.randrange(1, u) # select a random element from range(u)
        sub = set(random.sample(U, sub_size))
        L += [sub]


    # Create a dictionary to label L's elements with numbers.
    subsets = {i:s for i,s in enumerate(L)}


    if print_instance:
        # Print all the data.
        print('-'*80)
        print('Number of U\'s elements u =', u)
        print('Number of subsets s =', s)
        print('Maximum number that can appear in the sets n-1 =', n-1)
        print('U is:', U)
        print('The subsets are:', subsets)
        print('-'*80)


    if create_graph:
        # Create a graph where nodes are L's elements.
        plt.figure()

        G = nx.Graph()
        G.add_nodes_from(subsets)

        # While creating the edges make sure to get rid of self-edges (i!=j).
        keys = subsets.keys()
        G.add_edges_from([(i,j) for i in keys for j in keys if subsets[i]&subsets[j] and i!=j])

        nx.draw(G, with_labels=True, font_size=15, alpha=0.8)

    return U, subsets


# ********************************************************************************

def ComputeEnergy(x, U, subsets):
    """
    Arguments
    ---------
        x : a binary array.
        U : a set.
        subsets : a dictionary containing subsets of U, whose union is U.

    Return
    ------
        E : x's energy (See "Ising formulations of many NP problems" by Andrew Lucas)
    """


    A = 1. # constant parameter of H_A.
    B = 1. # constant parameter of H_B.

    E_A = 0.
    for uu in U:
        counts = sum([x[i] for i in subsets.keys() if uu in subsets[i]])
        #print(f'uu = {uu}, counts = {counts}')
        E_A += (1 - counts)**2

    E_A = A * E_A
    E_B = B * sum(x)

    E = E_A + E_B    

    return E


# ********************************************************************************

def BuildHam(x, U, subsets):
    """
    Arguments
    ---------
        x : a binary array.
        U : a set.
        subsets : the dictionary containing subsets of U, whose union is U.

    Return
    ------
        H : the Hamiltonian H of the problem.
            (See "Ising formulations of many NP problems" by Andrew Lucas)
        E : x's energy.
    """

    A = 1. # constant parameter of H_A.
    B = 1. # constant parameter of H_B.

    s = len(x) # number of subsets.
    
    H_A = np.zeros(shape=(s,s))
    for uu in U:
        for i in range(s):
            if uu in subsets[i]:
                H_A[i,i] += -1.
            for j in range(i+1,s):  
                if (uu in subsets[i] and uu in subsets[j]):
                    H_A[i,j] += 2.        

    H_A = A * H_A
    H_B = B * np.identity(s)

    H = H_A + H_B

    # Compute the expectation value <x|H|x> and remember to add the constant!
    E = np.dot(x, np.matmul(H,x)) + len(U)

    return H, E


# ********************************************************************************

def BitGen(n):
    """
    Arguments
    ---------
       n: length of the tuples to be produced.
       
    Return
    ------
       A generator of all possible tuples of n bits.
    """
    return itertools.product([0, 1], repeat=n)


# ********************************************************************************

def PlotEnergy(E_list, x_labels):
    """
    Arguments
    ---------
       E_list : a list of energies.
       x_labels : a list of strings, each representing 
                  the i_th state, whose energy is the i_th 
                  element in E_list.
      
    Return
    ------
       A plot of E_list as a function of x_labels.
    """
    fig, ax = plt.subplots(figsize=(19, 10))
    plt.title("Energy landscape")
    plt.subplots_adjust(left=0.03, bottom=0.2, right=0.94, top=0.94)
    xx = np.linspace(0,len(E_list),len(E_list))
    yy = E_list
    ax.plot(xx, yy, marker='o', linestyle='--', color='red')
    ax.set_ylabel("Energy")
    ax.set_xticks(xx)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.grid() 
    return


#**************************************************************************

if __name__ == '__main__':
    start_time = time.time()


    # Parameters.
    u = 20# size of U set
    s = 6 # number of subsets of U, whose union is U
    n = 30 # the maximum number that can appear in the sets will be n-1 
           # must be n >= u, as we are sampling without replacement
    len_x = s # length of x list, representing the state
    build_H = False


    # Randomly generate an instance of the problem.
    U, subsets = MEC_instance(u, s, n, print_instance=True, create_graph=True)


    # Define empty lists.
    E_list = [] # will contain the energies associated to each state
    x_labels = [] # will contain bit-strings describing the states (only needed for the plot)


    # Iterate over all possible states.
    for x in BitGen(len_x):

        x = np.array(x)
        x_labels.append(str(x))


        if build_H:

            # Build the Hamiltonian of the problem and compute x's energy.
            H, E = BuildHam(x, U, subsets)
            E_list.append(E)
            # print(f'\nH =\n{H}')
            # print(f'Energy (expectation value on H) = {E}')

        else: # this is faster

            # Given x, directly compute its energy.
            E = ComputeEnergy(x, U, subsets)
            E_list.append(E)
            # print(f'\nDirectly computed energy = {E}')


    # Plot energies.
    if(s<=10): # otherwise the computation is too heavy
        PlotEnergy(E_list, x_labels)


    # Find the minimum energy and the states that minimize it.
    E_min = min(E_list)
    x_labels = np.array(x_labels)
    x_min = x_labels[E_list == E_min]
    print(f"\n-> Minimum of energy: E_min={E_min}")
    print(f"-> States that minimize energy: {x_min}")


    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
    plt.show()
