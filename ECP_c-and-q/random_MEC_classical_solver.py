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
from typing import List, Dict, Set

from matplotlib import pyplot as plt
import numpy as np

from utils import build_instance, bit_gen, from_bool_to_numeric_lst

# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


# ********************************************************************************
def build_ham_MEC(x: np.ndarray, U: Set[int], subsets: Dict[int, Set[int]]) -> (np.ndarray, float):
    """
    Constructs the Hamiltonian and computes the energy for the Minimum Exact Cover (MEC) problem.

    The Hamiltonian is formulated based on the Ising model, as discussed in
    "Ising formulations of many NP problems" by Andrew Lucas.

    Parameters
    ----------
    x : numpy.ndarray
        A binary array representing the solution (variables are either 0 or 1).
    U : set of int
        A set of elements representing the universe in the Exact Cover problem.
    subsets : dictionary of {int: set of int}
        A dictionary where the keys are subset indices and the values are sets
        representing subsets of U.

    Returns
    -------
    H : numpy.ndarray
        The Hamiltonian matrix for the problem.
    E : float
        The energy of the configuration x.
    """

    B = 1.  # constant parameter of H_B.
    A = len(U) * (B + 1)  # constant parameter of H_A. It must be A > u*B

    s = len(x)  # number of subsets
    
    H_A = np.zeros(shape=(s, s))  # Initialize the Hamiltonian matrix
    for uu in U:
        for i in range(s):
            if uu in subsets[i]:
                H_A[i, i] += -1.  # Update diagonal elements
            for j in range(i + 1, s):  
                if (uu in subsets[i] and uu in subsets[j]):
                    H_A[i, j] += 2.  # Update off-diagonal elements

    H_A = A * H_A  # Apply the scaling factor to H_A
    H_B = B * np.identity(s)  # Identity matrix for H_B

    H = H_A + H_B  # Final Hamiltonian

    # Compute the expectation value <x|H|x> and add the constant term
    E = np.dot(x, np.matmul(H, x)) + len(U)

    return H, E


# ********************************************************************************
def compute_energy_MEC(x: np.ndarray, U: Set[int], subsets: Dict[int, Set[int]]) -> (float, float):
    """
    Computes the energy for the Minimum Exact Cover (MEC) problem.

    The energy is calculated using the Ising formulation for the Exact Cover problem,
    as described in "Ising formulations of many NP problems" by Andrew Lucas.

    Parameters
    ----------
    x : numpy.ndarray
        A binary array representing the solution (variables are either 0 or 1).
    U : set of int
        A set of elements representing the universe in the Exact Cover problem.
    subsets : dictionary of {int: set of int}
        A dictionary where the keys are subset indices and the values are sets
        representing subsets of U.

    Returns
    -------
    E_A : float
        The energy corresponding to H_A.
    E_B : float
        The energy corresponding to H_B.
    """

    B = 1.  # constant parameter of H_B.
    A = len(U) * (B + 1)  # constant parameter of H_A. It must be A > u*B

    E_A = 0.0
    for uu in U:
        counts = sum([x[i] for i in subsets.keys() if uu in subsets[i]])
        E_A += (1 - counts) ** 2  # Add contribution to energy for H_A

    E_A = A * E_A  # Apply scaling factor for H_A
    E_B = B * sum(x)  # Calculate energy for H_B

    return E_A, E_B


# ********************************************************************************
def plot_energy(E_list: List[float], x_list: List[List[int]]) -> None:
    """
    Plots the energy landscape of the system as a function of different configurations.

    The plot shows how the energy varies with the binary configurations in `x_list`.

    Parameters
    ----------
    E_list : list of float
        A list of energy values corresponding to different configurations.
    x_list : list of lists of int
        A list of binary configurations (represented as lists) whose energy values
        are stored in `E_list`.

    Returns
    -------
    None
        Displays a plot of the energy landscape.
    """

    fig, ax = plt.subplots(figsize=(19, 10))
    plt.title("Energy Landscape")
    plt.subplots_adjust(left=0.03, bottom=0.2, right=0.94, top=0.94)
    
    xx = np.linspace(0, len(E_list), len(E_list))  # X-axis values (index of configurations)
    yy = E_list  # Y-axis values (corresponding energies)
    
    ax.plot(xx, yy, marker='o', linestyle='--', color='red')
    ax.set_ylabel("Energy")
    ax.set_xticks(xx)
    
    # Set x-tick labels as string representations of the configurations
    x_labels = [str(x) for x in x_list]
    ax.set_xticklabels(x_labels, rotation=90)
    
    ax.grid()  # Display grid
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
