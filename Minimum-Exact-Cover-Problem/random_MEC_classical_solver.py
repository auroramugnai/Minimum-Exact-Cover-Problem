"""
Minimum exact cover.

    This script solves the Minimum Exact Cover (MEC) problem, as described in
    "Ising formulations of many NP problems" by Andrew Lucas. The MEC problem
    aims to find subsets from a given collection such that their union covers
    all elements exactly once while minimizing the energy of the configuration.

    - Randomly generates a problem instance, including a universal set `U` of
      size `u` with elements chosen from the range [0, n-1], and a collection of
      `s` subsets whose union equals `U`.

    - Builds a graph where each node represents a subset, and edges indicate
      overlapping subsets.

    - Iterates through all possible binary states representing subset inclusion
      to compute the energy of each state. Energy is computed either via the
      Hamiltonian matrix (if `build_H` is True) or directly (faster method).

    - Identifies and outputs the exact covers, their energies, and the minimum
      energy configuration(s).

    Notes:
        If `build_H` is set to True, the Hamiltonian matrix `H` of the problem
        is constructed, and energies are calculated using expectation values
        <x|H|x>. Otherwise, energies are computed directly, which is faster.

    Outputs:
        - Exact covers and their associated energies.
        - Minimum energy configuration(s) if exact covers exist.
        - Computation time and optional energy plots (for small problem sizes).
"""
import time

from matplotlib import pyplot as plt
import numpy as np

from utils import build_instance, bit_gen, from_bool_to_numeric_lst

# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


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

if __name__ == '__main__':
    start_time = time.time()


    # Parameters.
    u = 5 # size of U set
    s = 6 # number of subsets of U, whose union is U
    n = u # the maximum number that can appear in the sets will be n-1 
          # must be n >= u, as we are sampling without replacement
    len_x = s # length of x list, representing the state
    build_H = False

    # Randomly generate an instance of the problem.
    U, subsets = build_instance(u, s, n, print_instance=True, create_graph=False)

    # Define empty lists.
    E_list = [] # will contain the energies associated to each state (E_A + E_B)
    E_A_list = [] # we need this to see if the cover is exact 
    x_list = [] # will contain all the states x


    # Iterate over all possible states.
    for x in bit_gen(len_x):

        x = np.array(x, dtype=int)
        x_list.append(x)


        if build_H:

            # Build the Hamiltonian of the problem and compute x's energy.
            H, E = build_ham_MEC(x, U, subsets)
            E_list.append(E)
            # print(f'\nH =\n{H}')
            # print(f'Energy (expectation value on H) = {E}')

        else: # this is faster

            # Given x, directly compute its energy.
            E_A, E_B = compute_energy_MEC(x, U, subsets)
            E_list.append(E_A + E_B)
            E_A_list.append(E_A) 
            # print(f'\nDirectly computed energy = {E}')


    # Plot energies.
    if(s <= 10): # otherwise the computation is too heavy
        plot_energy(E_list, x_list)


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
        print("    -> Exact cover(s):", from_bool_to_numeric_lst(exact_covers))

        exact_covers_energies = np.array(E_list)[np.array(E_A_list) == 0.0]
        print("    -> Exact cover(s) energy:", exact_covers_energies)

        E_min = min(exact_covers_energies)
        print(f"    -> Energy minimum: E_min = {E_min}")

        minimum_exact_covers = exact_covers[exact_covers_energies == E_min]
        minimum_exact_covers = from_bool_to_numeric_lst(minimum_exact_covers)
        print("    -> Minimum exact cover(s):", minimum_exact_covers)


    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
    plt.show()
