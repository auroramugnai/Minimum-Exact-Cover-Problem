import itertools
import random
import pprint
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as pltù


# ********************************************************************************
def build_instance(u, s, n, print_instance, create_graph):
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
def bit_gen(n):
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
def from_bool_to_numeric_lst(bool_list: List[int]) -> List[int]:
    """
    Converts a boolean-like list to a list of indices where the values are 1.

    Parameters
    ----------
    bool_list : List[int]
        A list of integers (0 or 1), e.g., [0, 1, 1].

    Returns
    -------
    List[int]
        A list of indices where the values in the input list are 1.

    Examples
    --------
    >>> from_bool_to_numeric_lst([0, 1, 1])
    [1, 2]
    """
    return list(np.where(np.array(bool_list) == 1)[0])
