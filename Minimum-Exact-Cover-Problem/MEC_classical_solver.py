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

from utils import *

# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation




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
    U, subsets = MEC_instance(u, s, n, print_instance=True, create_graph=False)

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
        print("    -> Exact cover(s):", bool_states_to_num(exact_covers))

        exact_covers_energies = np.array(E_list)[np.array(E_A_list) == 0.0]
        print("    -> Exact cover(s) energy:", exact_covers_energies)

        E_min = min(exact_covers_energies)
        print(f"    -> Energy minimum: E_min = {E_min}")

        minimum_exact_covers = exact_covers[exact_covers_energies == E_min]
        minimum_exact_covers = bool_states_to_num(minimum_exact_covers)
        print("    -> Minimum exact cover(s):", minimum_exact_covers)


    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
    plt.show()
