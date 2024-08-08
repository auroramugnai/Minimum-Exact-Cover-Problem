import ast # to evaluate strings as arrays
import itertools
import random
import pprint
import time
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from termcolor import colored # to color a dataframe


# ********************************************************************************
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
def compute_energy(x, U, subsets):
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

    A = 1.

    E_A = 0.
    for uu in U:
        counts = sum([x[i] for i in subsets.keys() if uu in subsets[i]])
        #print(f'uu = {uu}, counts = {counts}')
        E_A += (1 - counts)**2

    E_A = A * E_A

    return E_A


# ********************************************************************************
def build_ham(U, subsets):
    """
    Arguments
    ---------
        U : set.
        subsets : dictionary containing subsets of U, whose union is U.

    Return
    ------
        H_A : Hamiltonian of the problem.
             (See "Ising formulations of many NP problems" by Andrew Lucas)
    """

    
    A = 1
    s = len(subsets) # number of subsets.
    
    H_A = np.zeros(shape=(s,s))
    for uu in U:
        for i in range(s):
            if uu in subsets[i]:
                H_A[i,i] += -1
            for j in range(i+1,s):  
                if (uu in subsets[i] and uu in subsets[j]):
                    H_A[i,j] += 2     

    H_A = A * H_A

    return H_A


# ********************************************************************************
def compute_energy_MEC(x, U, subsets):
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
def build_ham_MEC(x, U, subsets):
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
def energy_expect_val(H_A, x, u):
    """
    Arguments
    ---------
        H_A : Hamiltonian of the problem.
        x : binary array.
        u : length of U set.

    Return
    ------
        E_A : energy of x, computed as expectation value on H_A.
    """

    # Compute the expectation value <x|H_A|x> and remember to add the constant!
    E_A = np.dot(x, np.matmul(H_A,x)) + u

    return E_A

# ********************************************************************************
def plot_energy(E_list, x_list):
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


#**************************************************************************
def subsets_to_bool(U, subsets):
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
def bool_states_to_num(x_list):
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





def from_str_to_arr(x: str) -> List[int]:
    """
    Arguments
    ---------
        x (string of bits): such as "[0 0 1 0]"

    Return
    ------
        x (list of integers): such as [0, 0, 1, 0]
    """
    x = x.replace(" ", ", ")
    x = np.array(ast.literal_eval(x))
    x = list(np.where(x == 1)[0])

    return x


# ******************************************************************************

def get_counts(df: pd.DataFrame, col: str, print_df_counts: bool) -> pd.DataFrame:
    """
    Given a dataframe df, gets elements in the "col" column and 
    displays their occurrences in a new dataframe df_counts.

    Arguments
    ---------
        df (pandas Dataframe): 
            dataframe from which to extract "col" column.
        col (string):
            name of the column whose elements counts we're interested in.
        print_df_counts (bool):
            if True, prints the counts'dataframe.

    Return
    ------
        df_counts (pandas Dataframe):
            dataframe containing col's elements with their counts,
            sorted in descending order.

    """

    df_counts = df.copy(deep=False)

    # Group by the 'array' column and count the occurrences.
    df_counts = df_counts[["array"]].groupby(["array"])["array"].count()
    df_counts = df_counts.reset_index(name='counts')

    # Sort in descending order.
    df_counts = df_counts.sort_values(['counts'], ascending=False)
    df_counts = df_counts.reset_index(drop=True)

    if print_df_counts:
        print(f"\n-- counts dataframe:\n", df_counts.head())

    return df_counts


# ******************************************************************************

def find_U_from_subsets(subsets_dict):
    U = subsets_dict[0]
    for s in subsets_dict.values():
        U = U | s
    return U


# ******************************************************************************

def is_feasible(state, subsets):
    """ Checks if a state selects subsets that have 0 intersection 
        (= if it is feasible).

        Parameters
        ----------
            state (str): the state to be checked.
            subsets (list or dict): a list containing the subsets of the problem,
                                    or a dictionary where keys are natural numbers 
                                    and values are the subsets of the problem.

        Returns
        -------
            check (bool): True if the state is feasible, False otherwise.

    """

    state = ast.literal_eval(state) # str -> list

    chosen_subsets = [subsets[int(i)] for i in state]
    union_set = set().union(*chosen_subsets)
    sum_of_len = sum([len(sub) for sub in chosen_subsets])

    """ The sum of the lengths of the subsets selected by a state is 
        equal to the length of the union set `union_set` only if the 
        subsets do not intersect, that is, if the state is feasible.
    """

    if len(union_set) == sum_of_len:
        check = True
    else:
        check = False

    return check
