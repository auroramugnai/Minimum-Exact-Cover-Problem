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
    - (Optional) Plots the energy landscape for visualization.
    - Includes placeholders for solving the problem using D-Wave and other QUBO
      solvers, although these are currently commented out.

    Outputs:
    - Exact covers found through the QUBO approach.
    - Verification results using a library-based method.
    - Computation time and optional plots for small problem instances.
"""

from datetime import datetime
import time

import exact_cover as ec

from utils import *

# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


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
    U, subsets = MEC_instance(u, s, n, print_instance=True, create_graph=False)

    # ---------------------------------------------------------------------------
    # U = {0, 1, 2, 3, 4, 5}
    # u = len(U)
    # subsets = {0: {0, 1, 2, 3, 5},
    #            1: {1, 5},
    #            2: {2, 3, 5},
    #            3: {3, 4},
    #            4: {1, 2, 3, 5},
    #            5: {4},
    #            6: {0, 1},
    #            7: {0},
    #            8: {1},
    #            9: {0, 1, 2, 4, 5}}
    # s = len(subsets)
    # len_x = s

    # SOLUTIONS = [[4, 5, 7], [2, 5, 7, 8], [2, 5, 6], [0, 5]]
    # print("\nTRUE SOLUTIONS: ", SOLUTIONS)
    # ---------------------------------------------------------------------------
   
    # Build the Hamiltonian of the problem
    H_A = build_ham(U, subsets)
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


    # Plot energies.
    if(s <= 10): # otherwise the computation is too heavy
        PlotEnergy(E_list, x_list)


    print("\n")
    print('*'*30, "SOLUTION", '*'*30)
    print("\nQUBO SOLUTION:")


    # Find the exact covers.
    exact_covers = np.array(x_list)[np.array(E_A_list) == 0.0]

    if len(exact_covers):
        print("    -> Exact cover(s):", bool_states_to_num(exact_covers))

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
