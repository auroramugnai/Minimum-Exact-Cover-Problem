"""
Exact Cover Solver.
(See "Ising formulations of many NP problems" by Andrew Lucas)

    This script solves the Exact Cover problem using multiple approaches,
    including QUBO formulation and a Python library method. The Exact Cover
    problem aims to select subsets from a collection such that their union 
    exactly matches a given universal set `U`.

    The script performs the following steps:

    - Randomly generates a problem instance with:
        - `U`: a set of `u` elements chosen arbitrarily between 0 and `n-1`.
        - `subsets`: a dictionary containing `s` subsets of `U` whose union equals `U`.

    - Constructs the Hamiltonian matrix `H_A` for the QUBO representation of the
      problem and (optionally) expands it for multi-unit systems.

    - Iterates over all possible binary states representing subset inclusion,
      computing energies for each state either as expectation values `<x|H_A|x>`
      or directly.

    - Identifies exact covers (states with zero energy) and outputs their details.

    - Uses a Python library to solve the Exact Cover problem as a validation 
      step, displaying the exact cover(s) and the total number of solutions.

    Additional features:
    - Includes placeholders for solving the problem using D-Wave and other QUBO
      solvers, although these are currently commented out.

    Outputs:
    - Exact covers found through the QUBO approach.
    - Verification results using a library-based method.
    - Computation time and optional plots for small problem instances.
"""

from datetime import datetime
import time
from typing import Set, Dict, List

from matplotlib import pyplot as plt
import numpy as np

import exact_cover as ec
from utils import build_instance, bit_gen, from_bool_to_numeric_lst


# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


# ********************************************************************************
def build_ham_EC(U: Set[int], subsets: Dict[int, Set[int]]) -> np.ndarray:
    """
    Constructs the Hamiltonian matrix for the Exact Cover problem.

    The Hamiltonian is formulated based on the Ising model, as discussed in
    "Ising formulations of many NP problems" by Andrew Lucas.

    Parameters
    ----------
    U : set of int
        A set of elements, typically representing the universe in the Exact Cover problem.
    subsets : dictionary of {int: set of int}
        A dictionary where the keys are subset indices and the values are sets
        representing subsets of the universe U.

    Returns
    -------
    H_A : numpy.ndarray
        The Hamiltonian matrix for the problem.
    """

    A = 1
    s = len(subsets)  # number of subsets
    
    H_A = np.zeros(shape=(s, s))  # Initialize the Hamiltonian matrix
    for uu in U:
        for i in range(s):
            if uu in subsets[i]:
                H_A[i, i] += -1
            for j in range(i + 1, s):  
                if (uu in subsets[i] and uu in subsets[j]):
                    H_A[i, j] += 2     

    H_A = A * H_A  # Apply the scalar factor to the Hamiltonian matrix

    return H_A


# ********************************************************************************
def compute_energy(x: np.ndarray, U: Set[int], subsets: Dict[int, Set[int]]) -> float:
    """
    Computes the energy of a given binary configuration.

    The energy is calculated using the Ising formulation for the Exact Cover problem,
    as described in "Ising formulations of many NP problems" by Andrew Lucas.

    Parameters
    ----------
    x : numpy.ndarray
        A binary array representing the solution (variables are either 0 or 1).
    U : set of int
        A set representing the universe of elements in the Exact Cover problem.
    subsets : dictionary of {int: set of int}
        A dictionary of subsets of U, where the keys are subset indices and
        the values are sets representing the subsets.

    Returns
    -------
    E_A : float
        The energy of the configuration x.
    """

    A = 1.0  # Scaling factor for the energy

    E_A = 0.0
    for uu in U:
        counts = sum([x[i] for i in subsets.keys() if uu in subsets[i]])
        E_A += (1 - counts) ** 2  # Add contribution to energy

    E_A = A * E_A  # Apply scaling factor

    return E_A


# ********************************************************************************
def energy_expect_val(H_A: np.ndarray, x: np.ndarray, u: int) -> float:
    """
    Computes the expected energy of a binary configuration using the Hamiltonian.

    The expectation value is calculated as <x|H_A|x> + constant, where H_A is
    the Hamiltonian matrix and x is the binary configuration.

    Parameters
    ----------
    H_A : numpy.ndarray
        The Hamiltonian matrix for the problem.
    x : numpy.ndarray
        A binary array representing the solution (variables are either 0 or 1).
    u : int
        The length of the set U (number of elements in the universe).

    Returns
    -------
    E_A : float
        The expected energy of the configuration x.
    """

    # Compute the expectation value <x|H_A|x> and add the constant term.
    E_A = np.dot(x, np.matmul(H_A, x)) + u

    return E_A


# ********************************************************************************
def subsets_to_bool(U: Set[int], subsets: Dict[int, Set[int]]) -> List[np.ndarray]:
    """
    Converts each subset of U into its boolean representation.

    For each subset, creates a binary vector where each position corresponds
    to an element in U, and the value is 1 if the element is in the subset, 
    otherwise 0.

    Parameters
    ----------
    U : set of int
        A set representing the universe of elements.
    subsets : dictionary of {int: set of int}
        A dictionary where the keys are subset indices and the values are sets
        representing the subsets of U.

    Returns
    -------
    bool_subsets : list of numpy.ndarray
        A list of binary arrays representing the subsets in boolean form.
    """

    bool_subsets = []
    for subset in subsets.values():
        bool_subset = np.zeros(len(U), dtype=bool)  # Initialize a boolean array
        for i, uu in enumerate(U):
            if uu in subset:
                bool_subset[i] = 1  # Set the corresponding position to 1
        bool_subsets.append(bool_subset)

    return bool_subsets



if __name__ == '__main__':
    start_time = time.time()

    # get current date and time
    current_datetime = datetime.now().strftime("@%Y-%m-%d@%Hh%Mm%Ss")

    # Parameters.
    u = 6 # size of U set
    s = 10 # number of subsets of U, whose union is U
    n = u # the maximum number that can appear in the sets will be n-1 
          # must be n >= u, as we are sampling without replacement
    len_x = s # length of x list, representing the state


    # Randomly generate an instance of the problem.
    U, subsets = build_instance(u, s, n, print_instance=True, create_graph=False)

    # ---------------------------------------------------------------------------
   
    # Build the Hamiltonian of the problem
    H_A = build_ham_EC(U, subsets)
    # print("H_A = \n", H_A)


    # Create a bigger Hamiltonian by copying H_A over num_units subspaces.
    num_units = 15
    big_I = np.eye(num_units, dtype=int)
    big_H_A = np.kron(big_I, H_A)
    # print("big_H_A = \n", big_H_A)


    # Define empty lists.
    E_A_list = [] # energies
    x_list = [] # states


    # Iterate over all possible states.
    for x in bit_gen(len_x):

        x = np.array(x, dtype=int)
        x_list.append(x)

        # Compute x's energy.
        E_A = energy_expect_val(H_A, x, u)
        E_A_list.append(E_A)

        # (Optional) Just to double check.
        if E_A != compute_energy(x, U, subsets):
            print("ERROR: expectation value on H_A != straightforwardly computed energy")



    print("\n")
    print('*'*30, "SOLUTION", '*'*30)
    print("\nQUBO SOLUTION:")


    # Find the exact covers.
    exact_covers = np.array(x_list)[np.array(E_A_list) == 0.0]

    if len(exact_covers):
        print("    -> Exact cover(s):", from_bool_to_numeric_lst(exact_covers))

    else:
        print("There is no exact cover")
        # Exit with success
        # import sys
        # sys.exit(0) 


    
    # **********************************************************************
    
    print("\nPYTHON LIBRARY SOLUTION:")
    """
    Let's look for a solution without the QUBO.

    First, rewrite subsets as lists of length len(U)=u of bits. 
    The list corresponding to a set S will have '1' 
    in the i-th position if the i-th element of U is in S.
    """

    bool_subsets = np.array(subsets_to_bool(U, subsets))

    exact_cover = ec.get_exact_cover(bool_subsets)
    print(f"    -> Exact cover:{np.sort(exact_cover)}")
    num_exact_covers = ec.get_solution_count(bool_subsets)
    print(f"    -> Number of exact covers: {num_exact_covers}")

   
    # ---------------------------------------------------------------------  

    # # Print the small Hamiltonian to file.
    # with open(f"{PROBLEM_NAME}_u={u}_s={s}_{current_datetime}.txt",'wb') as f:
        
    #     mat = np.matrix(H_A)
    #     for line in mat:
    #         np.savetxt(f, line, fmt='%d')


    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
    plt.show()
