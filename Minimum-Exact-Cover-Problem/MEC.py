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
import pprint

import exact_cover as ec
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

    while(len(L) != s-1):
        sub_size = random.randrange(1, u) # select a random element from range(u)
        sub = set(random.sample(U, sub_size))

        # If sub is already in L, repeat the i-th step.
        if any(sub == x for x in L): 
            print("INFO: Two equal subsets were randomly generated. Re-extracting one subset...")
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
        print("\n")
        print('*'*30, "INSTANCE", '*'*30)
        print("\n")
        print('- Number of U\'s elements u =', u)
        print('- Number of subsets s =', s)
        print('- Maximum number that can appear in the sets n-1 =', n-1)
        print('- U is:', U)
        print('- The subsets are:')
        pprint.pprint(subsets)


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


    B = 1. # constant parameter of H_B.
    A = len(U) * (B+1) # constant parameter of H_A. It must be A > u*B

    E_A = 0.
    for uu in U:
        counts = sum([x[i] for i in subsets.keys() if uu in subsets[i]])
        #print(f'uu = {uu}, counts = {counts}')
        E_A += (1 - counts)**2

    E_A = A * E_A
    E_B = B * sum(x)


    return E_A, E_B


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

    
    B = 1. # constant parameter of H_B.
    A = len(U) * (B+1) # constant parameter of H_A. It must be A > u*B

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
       A generator of all possible n-tuples of bits.
    """
    return itertools.product([0, 1], repeat=n)


# ********************************************************************************

def PlotEnergy(E_list, x_list):
    """
    Arguments
    ---------
       E_list : a list of energies.
       x_list : a list of binary lists representing 
                the states whose energy is in E_list
      
    Return
    ------
       A plot of E_list as a function of x_list.
    """
    fig, ax = plt.subplots(figsize=(19, 10))
    plt.title("Energy landscape")
    plt.subplots_adjust(left=0.03, bottom=0.2, right=0.94, top=0.94)
    xx = np.linspace(0,len(E_list),len(E_list))
    yy = E_list
    ax.plot(xx, yy, marker='o', linestyle='--', color='red')
    ax.set_ylabel("Energy")
    ax.set_xticks(xx)
    x_labels = [str(x) for x in x_list]
    ax.set_xticklabels(x_labels, rotation=90)
    ax.grid() 
    return


#**************************************************************************

def SubsetsToBool(U, subsets):
    """
    Rewrite each subset in the boolean way. 
    For example, if U = {0,1,2}, then S = {0,2} becomes S = [1,0,1].

    Arguments
    ---------
        U (set): a set.
        subsets (dictionary): a dictionary whose keys are natural 
                              numbers, whose values are subsets of U

    Return
    ------
        bool_subsets (list of lists): the list of subsets written 
                                      in the boolean way.
    """
    bool_subsets = []
    for subset in subsets.values():
        bool_subset = np.zeros(len(U)).astype(bool)
        for i,uu in enumerate(U):
            if(uu in subset):
                bool_subset[i] = 1
        bool_subsets.append(bool_subset)

    return bool_subsets


#**************************************************************************

def StatesFromBoolToNum(x_list):
    """
    The "1" in the i-th position of a bit-list describing 
    a state x means that the i-th subset in the 
    "subsets" dictionary is selected. We thus can rewrite 
    the bit-list x as a list of natural numbers that label 
    the subsets selected.
    Example: if subsets = {0: {0,1}, 1:{1,2}, 2:{0,2}}
             then x = [1 0 1] becomes x = [0,2]

    Arguments
    ---------
        x_list: a list of bit-lists

    Return
    ------
        x_numbers_list: list of natural-numbers-lists
    """
    
    x_numbers_list = []
    for x in x_list:
        subsets_selected = list(np.where(x == 1)[0])
        x_numbers_list.append(subsets_selected)

    return x_numbers_list



#**************************************************************************

if __name__ == '__main__':
    start_time = time.time()


    # Parameters.
<<<<<<< HEAD
    u = 5 # size of U set
=======
    u = 20 # size of U set
>>>>>>> 65a29f8f81742e4b21607ad9a01e2bc629524c98
    s = 6 # number of subsets of U, whose union is U
    n = u # the maximum number that can appear in the sets will be n-1 
          # must be n >= u, as we are sampling without replacement
    len_x = s # length of x list, representing the state
    build_H = False

    # Randomly generate an instance of the problem.
    U, subsets = MEC_instance(u, s, n, print_instance=True, create_graph=False)

    # Define empty lists.
    E_list = [] # will contain the energies associated to each state (E_A + E_B)
    E_A_list = [] # we need this to see if the cover is exact 
    x_list = [] # will contain all the states x


    # Iterate over all possible states.
    for x in BitGen(len_x):

        x = np.array(x, dtype=int)
        x_list.append(x)


        if build_H:

            # Build the Hamiltonian of the problem and compute x's energy.
            H, E = BuildHam(x, U, subsets)
            E_list.append(E)
            # print(f'\nH =\n{H}')
            # print(f'Energy (expectation value on H) = {E}')

        else: # this is faster

            # Given x, directly compute its energy.
            E_A, E_B = ComputeEnergy(x, U, subsets)
            E_list.append(E_A + E_B)
            E_A_list.append(E_A) 
            # print(f'\nDirectly computed energy = {E}')


    # Plot energies.
    if(s <= 10): # otherwise the computation is too heavy
        PlotEnergy(E_list, x_list)


    print("\n")
    print('*'*30, "SOLUTION", '*'*30)
    print("\nQUBO SOLUTION:")


    # Find the exact covers.
    exact_covers = np.array(x_list)[np.array(E_A_list) == 0.0]

    if(exact_covers.size == 0):
        print("There is no exact cover")
        # Exit with success
        # import sys
        # sys.exit(0) 

    else:
        print("    -> Exact cover(s):", StatesFromBoolToNum(exact_covers))

        exact_covers_energies = np.array(E_list)[np.array(E_A_list) == 0.0]
        print("    -> Exact cover(s) energy:", exact_covers_energies)

        E_min = min(exact_covers_energies)
        print(f"    -> Energy minimum: E_min = {E_min}")

        minimum_exact_covers = exact_covers[exact_covers_energies == E_min]
        minimum_exact_covers = StatesFromBoolToNum(minimum_exact_covers)
        print("    -> Minimum exact cover(s):", minimum_exact_covers)


    """
    **********************************************************************
    Let's look for a solution without the QUBO.

    First, rewrite subsets as lists of length len(U)=u of bits. 
    The list corresponding to a set S will have '1' 
    in the i-th position if the i-th element of U is in S.
    """

    bool_subsets = np.array(SubsetsToBool(U, subsets))

    print("\nPYTHON LIBRARY SOLUTION:")
    exact_cover = ec.get_exact_cover(bool_subsets)
    print(f"    -> Exact cover:{np.sort(exact_cover)}")
    num_exact_covers = ec.get_solution_count(bool_subsets)
    print(f"    -> Number of exact covers: {num_exact_covers}")

    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
    plt.show()
