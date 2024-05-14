"""Same as EC.py, but with an instance fixed.
"""

from datetime import datetime
import time

from utils import *

# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


if __name__ == '__main__':
    start_time = time.time()

    # get current date and time
    current_datetime = datetime.now().strftime("@%Y-%m-%d@%Hh%Mm%Ss")

    # ---------------------------------------------------------------------------
    U = {0, 1, 2, 3, 4, 5}
    u = len(U)
    subsets = {0: {0, 1, 2, 3, 5},
               1: {1, 5},
               2: {2, 3, 5},
               3: {3, 4},
               4: {1, 2, 3, 5},
               5: {4},
               6: {0, 1},
               7: {0},
               8: {1},
               9: {0, 1, 2, 4, 5}}
    s = len(subsets)
    len_x = s

    SOLUTIONS = [[4, 5, 7], [2, 5, 7, 8], [2, 5, 6], [0, 5]]
    print("\nTRUE SOLUTIONS: ", SOLUTIONS)
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


    # # Plot energies.
    # if(s <= 10): # otherwise the computation is too heavy
    #     PlotEnergy(E_list, x_list)


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

    # print("\nDWAVE SOLUTION (SteepestDescentSolver):")
    # from dwave.samplers import SteepestDescentSolver
    # solver = SteepestDescentSolver()
    # sampleset = solver.sample_qubo(big_H_A)
    # print(sampleset)

    # ---------------------------------------------------------------------

    # print("\nDWAVE SOLUTION (ExactSolver):")
    # from dimod import ExactSolver
    # sampler = ExactSolver()
    # sampleset = sampler.sample_qubo(big_H_A)
    # print(sampleset)

    # ---------------------------------------------------------------------

    # print("\nDWAVE SOLUTION (SimulatedAnnealingSampler):")
    # import neal
    # solver = neal.SimulatedAnnealingSampler()
    # sampleset = solver.sample_qubo(big_H_A, num_reads=100)
    #
    # df = sampleset.to_pandas_dataframe()
    # df.to_csv(f"./SimulatedAnnealing_{datetime}.csv", index=False)
    # print(sampleset)

    # ---------------------------------------------------------------------

    # print("\nDWAVE SOLUTION (DWaveSampler):")
    # from dwave.system import DWaveSampler, EmbeddingComposite
    # import dwave.inspector

    # NSAMPLES = 1
    # NREADS = 1

    # sampler = EmbeddingComposite(DWaveSampler(solver=dict(topology__type='zephyr')))
    # sampler_v = 2

    # PROBLEM = big_H_A
    PROBLEM_NAME = f'ExactCover_{num_units}units'

    # if sampler_v == 2:
    #     AdvVERSION = 'Adv2'
    # else:
    #     AdvVERSION = 'Adv1'

    # # Create a pandas DataFrame and save it to .csv.
    # for ith_sample in range(1, NSAMPLES+1):
    #     header = f'{PROBLEM_NAME}_{AdvVERSION}_{NREADS}_{ith_sample}of{NSAMPLES}'
    #     sampleset = sampler.sample_qubo(PROBLEM, num_reads=NREADS, label=header)
    #     df = sampleset.to_pandas_dataframe()
    #     df.to_csv(header + f'{current_datetime}.csv', index=False)

    #     print(sampleset)
    #     dwave.inspector.show(sampleset)

    # # import subprocess
    # # # run postprocess file
    # # subprocess.run(["python3", "EC_postprocess.py"])
              

    # ---------------------------------------------------------------------  

    # Print the small Hamiltonian to file.
    with open(f"{PROBLEM_NAME}_u={u}_s={s}_{current_datetime}.txt",'wb') as f:
        
        mat = np.matrix(H_A)
        for line in mat:
            np.savetxt(f, line, fmt='%d')


    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
    plt.show()
