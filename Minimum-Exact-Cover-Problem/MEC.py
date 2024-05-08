"""Minimum exact cover.

    (See "Ising formulations of many NP problems" by Andrew Lucas)

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
