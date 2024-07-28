"""Exact Cover.

   (See "Ising formulations of many NP problems" by Andrew Lucas)

    - Given n, u, s, return a set U of u random elements, 
      chosen arbitrarily between 0 and n-1, and a dictionary
      containing s subsets of U, whose union is U.

    - Create the graph representing the problem, where an edge between 
      nodes i,j means that the subsets corresponding to i,j have 
      a non-zero intersection.
    
    - Build the hamiltonian matrix H_A of the problem and compute
      energies as expectation values <x|H_A|x>.

    - (Optional) plot the energy landscape.

    - Find the exact cover (states with zero energy) iterating over
      all possible states.
"""

from datetime import datetime
import time

import pandas as pd
from utils import *


# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


if __name__ == '__main__':
    start_time = time.time()

    # get current date and time
    current_datetime = datetime.now().strftime("@%Y-%m-%d@%Hh%Mm%Ss")

    # ---------------------------------------------------------------------------    
    
    U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    subsets = {0: {2, 4, 5, 6, 9}, 
               1: {16, 13, 14, 15}, 
               2: {1, 4, 5, 6}, 
               3: {8, 10, 12, 7}, 
               4: {1, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16}, 
               5: {11, 9, 2, 3}, 
               6: {3, 4, 5, 7, 8, 10, 12}, 
               7: {3, 7, 8, 9, 10, 11, 14}
               }
    u = len(U)
    s = len(subsets)
    len_x = s

    SOLUTIONS = [[0, 4], [1, 2, 3, 5]]
    print("\nTRUE SOLUTIONS: ", SOLUTIONS)
    # ---------------------------------------------------------------------------


    # Build the Hamiltonian of the problem
    H_A = build_ham(U, subsets)
    # print("H_A = \n", H_A)


    # Create a bigger Hamiltonian by copying H_A over num_units subspaces.
    NUNITS = 30
    big_I = np.eye(NUNITS, dtype=int)
    big_H_A = np.kron(big_I, H_A)
    # print("big_H_A = \n", big_H_A)


  
    # ************************************************************************
    # *************************** DWAVE SAMPLING *****************************
    # ************************************************************************

    

    PROBLEM_NAME = f'EC_u{u}_s{s}_{NUNITS}units'

    # ------------------------- SIMULATED ANNEALING ------------------------ 

    # print("\nDWAVE SOLUTION (SimulatedAnnealingSampler):")
    # import neal
    # solver = neal.SimulatedAnnealingSampler()
    # NREADS = 100
    # sampleset = solver.sample_qubo(big_H_A, num_reads=NREADS)
    
    # df = sampleset.to_pandas_dataframe()
    # csv_path = f"./{PROBLEM_NAME}_{NREADS}_{current_datetime}.csv"
    # df.to_csv(csv_path, index=False)
    # print(sampleset)

    # --------------------------- QPU ---------------------------------------

    print("\nDWAVE SOLUTION (DWaveSampler):")
    from dwave.system import DWaveSampler, EmbeddingComposite
    import dwave.inspector

    NSAMPLES = 5
    NREADS = 100

    sampler = EmbeddingComposite(DWaveSampler(solver=dict(topology__type='zephyr')))
    sampler_v = 2

    if sampler_v == 2:
        AdvVERSION = 'Adv2'
    else:
        AdvVERSION = 'Adv1'

    # Create a pandas DataFrame and save it to .csv.
    for ith_sample in range(1, NSAMPLES+1):
        header = f'{PROBLEM_NAME}_{NREADS}_{AdvVERSION}_{ith_sample}of{NSAMPLES}'
        sampleset = sampler.sample_qubo(PROBLEM, NREADS=NREADS, label=header)
        df = sampleset.to_pandas_dataframe()
        csv_path = header + f'{current_datetime}.csv'
        df.to_csv(csv_path, index=False)

        print(sampleset)
        dwave.inspector.show(sampleset)



    # ************************************************************************
    # *************************** POSTPROCESSING *****************************
    # ************************************************************************


    num_digits = s*NUNITS # length of a sample 

    # --------------------------------------------------------------------------
    # --------------------   display the global dataframe   --------------------

    # df_tot is the dataframe containing the whole information of one run.
    df_tot = pd.read_csv(csv_path)

    # Add a column to represent states as arrays.
    df_tot["array"] = list(df_tot.iloc[:, 0:num_digits].to_numpy(dtype=int))


    # Drop samples' columns (all the info is now in "array" column).
    # df_tot.drop(df_tot.iloc[:, 0:num_digits], inplace=True, axis=1)

    # Turn the arrays to strings (arrays are not hashable).
    df_tot["array"] =  df_tot["array"].map(str)

    # print("\n--df_tot:\n", df_tot.head(10))


    # --------------------        count occurrences         --------------------
    
    df_counts = get_counts(df_tot, col="array", print_df_counts=False)



    # --------------------   create each unit's dataframe   --------------------
    # -----------------  & count occurrences (for each unit)  ------------------
    

    # Name df_unit's columns with numbers from 0 to s-1
    column_names = ['{}'.format(n) for n in range(s)]

    # Set how to split df_tot's columns.
    start_points = np.arange(0, num_digits, s)
    end_points = np.arange(s, num_digits+s, s)


    df_counts_list = []


    # ---------      Create a dataframe for each unit.      --------------------

    for i, (start, end) in enumerate(zip(start_points, end_points)):

        # Take df_tot's columns from "start" to "end". 
        df_unit = df_tot.iloc[:, start:end]
        df_unit.columns = column_names

        # Add a column so to represent states as arrays
        df_unit["array"] = list(df_unit.iloc[:].to_numpy())

        # Turn the arrays to strings (arrays are not hashable)
        df_unit["array"] =  df_unit["array"].map(str)

        # print(f"\n******* Unit number {i} *******\n", df_unit.head())

        # Get the counts.
        df_counts = get_counts(df_unit, col="array", print_df_counts=False)

        # For readibility, go from boolean language to numerical language.
        df_counts["array_to_num"] = df_counts["array"].apply(from_str_to_arr)
        df_counts["array_to_num"] = df_counts["array_to_num"].map(str)
        df_counts.drop(columns = "array", inplace=True)


        # Add unit-dataframe to the list of all unit dataframes.
        df_counts_list.append(df_counts)

        # print(df_counts)



    # ---------      Sum counts from each unit dataframe    --------------------

    df_sum = pd.concat(df_counts_list).groupby(['array_to_num']).sum()
    df_sum = df_sum.sort_values(['counts'], ascending=False).reset_index()
    print(f"\n******* Sum of the counts of each unit's dataframe *******")
    # print(df_sum.head())


    #-----------------        Print it with colors         --------------------

    df_print = df_sum.copy()

    # Define a function that colors arrays in green if they are a solution.
    color_solutions = lambda x: (colored(str(x), None, 'on_green') 
                                 if eval(x) in SOLUTIONS 
                                 else colored(x, 'white', None))

    df_print['array_to_num'] = df_print['array_to_num'].map(color_solutions)

    # Reset columns or they will be shifted.
    df_print.columns =  [colored('array_to_num', 'white', None)] + ["counts"]
    print(df_print)    

    #-----------------------------------------------------------------------------

    # Count the total number of correct solutions.
    correct = df_sum.loc[df_sum["array_to_num"].map(eval).isin(SOLUTIONS), "counts"].sum()
    total = NREADS * NUNITS
    print(f"\nACCURACY = correct / total = {correct}/{total} = {round((correct/total) * 100, 2)}%")
    print(f"Z = correct / NREADS = {correct}/{NREADS} = {round((correct/NREADS), 2)}")


    print(csv_path)
    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
    plt.show()
