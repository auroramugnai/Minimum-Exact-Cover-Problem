"""
Exact Cover Problem Solver.

This module implements a solver for the Exact Cover problem, leveraging Hamiltonian-based formulations
and quantum-inspired approaches.

Features:
---------
1. **Input Generation:**
   - Accepts user-defined parameters to generate:
     - A universal set `U` consisting of random integers.
     - Subsets of `U` forming the basis of the Exact Cover problem.

2. **Hamiltonian Construction:**
   - Constructs the Hamiltonian for the Exact Cover problem.
   - Extends the Hamiltonian to accommodate multiple units for quantum sampling.

3. **Quantum Sampling:**
   - Utilizes D-Wave's quantum annealer (Zephyr topology) to sample potential solutions.
   - Processes multiple samples to find solutions.

4. **Postprocessing:**
   - Aggregates sampled states across units and counts occurrences.
   - Labels states as Minimum Exact Cover (MEC), Exact Cover (EC), feasible, or invalid.
   - Calculates accuracy metrics for MEC and EC.

5. **Visualization and Reporting:**
   - Provides colored output to highlight solutions in the aggregated data.
   - Exports data to CSV for reproducibility.
   - Visualizes accuracy metrics through plots and saves results as JSON.

6. **User Input Handling:**
   - Dynamically accepts configurable parameters, such as the number of reads, samples, and units.
"""


from datetime import datetime
import json
import os
import time
from typing import List, Union, Dict, Set  # For type annotations
import sys

import numpy as np
import pandas as pd
from termcolor import colored # For coloring a dataframe

from random_EC_classical_solver import build_ham_EC
from instances import all_instances, all_solutions
from utils import from_bool_to_numeric_lst

# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation


# ******************************************************************************

# get current date and time
current_datetime = datetime.now().strftime("%m-%d@%Hh%Mm%Ss")

# Create a directory named with the current datetime
output_dir = f"./{current_datetime}"
os.makedirs(output_dir, exist_ok=True)
print(f"... Saving results in {output_dir} directory ... ")


# ****************************** FUNCTIONS **************************************

def find_U_from_subsets(subsets: Union[List[Set[int]], Dict[int, Set[int]]]) -> Set[int]:
    """
    Computes the union of all subsets in the given collection.

    Parameters
    ----------
    subsets : list of sets or dict of sets
        A collection of subsets. If a list, each element is a subset of integers.
        If a dictionary, keys are integers, and values are subsets.

    Returns
    -------
    set
        The union of all subsets in the input collection.

    Examples
    --------
    >>> find_U_from_subsets([{1, 2}, {2, 3}, {4}])
    {1, 2, 3, 4}

    >>> find_U_from_subsets({0: {1, 2}, 1: {2, 3}, 2: {4}})
    {1, 2, 3, 4}
    """
    U = set()
    for s in subsets.values() if isinstance(subsets, dict) else subsets:
        U |= s
    return U

def is_feasible(state: str, subsets: Union[List[Set[int]], Dict[int, Set[int]]]) -> bool:
    """
    Checks if a state selects subsets that do not intersect (i.e., it is feasible).

    Parameters
    ----------
    state : str
        A string representing a list of indices of subsets, e.g., "[0, 1, 2]".
    subsets : list of sets or dict of sets
        A collection of subsets. If a list, each element is a subset of integers.
        If a dictionary, keys are integers, and values are subsets.

    Returns
    -------
    bool
        True if the selected subsets do not intersect; otherwise, False.

    Examples
    --------
    >>> is_feasible("[0, 1]", [{1, 2}, {3, 4}, {2, 3}])
    True

    >>> is_feasible("[0, 2]", [{1, 2}, {3, 4}, {2, 3}])
    False
    """
    state_indices = list(map(int, state.strip("[]").split(",")))
    chosen_subsets = [subsets[i] for i in state_indices]
    union_set = set().union(*chosen_subsets)
    sum_of_lengths = sum(len(sub) for sub in chosen_subsets)
    return len(union_set) == sum_of_lengths

def get_input(prompt: str, default_value: int, type_func=int) -> int:
    """
    Prompts the user for input with a default value and converts the input to the specified type.

    Parameters
    ----------
    prompt : str
        The message displayed to the user as the input prompt.
    default_value : int
        The value returned if the user does not provide any input (presses Enter).
    type_func : callable, optional
        A function to convert the user's input to the desired type. Defaults to `int`.

    Returns
    -------
    int
        The user's input converted using `type_func`, or the `default_value` if no input is provided.
    """
    user_input = input(f"{prompt} (default {default_value}): ").strip()
    return type_func(user_input) if user_input else default_value

def from_str_to_list(state_as_str: str) -> List[int]:
    """
    Converts a string representation of a list to a list of integers.

    Parameters
    ----------
    state_as_str : str
        A string representation of a list of integers, e.g., "[1, 0, 1]".

    Returns
    -------
    List[int]
        A list of integers extracted from the input string.
    """
    return list(map(int, state_as_str.strip("[]").replace(",", "").split()))

def from_bool_to_numeric_str(bool_str: str) -> str:
    """
    Converts a string representation of a boolean-like list to a string of indices.

    Parameters
    ----------
    bool_str : str
        A string representation of a boolean-like list, e.g., "[0, 1, 1]".

    Returns
    -------
    str
        A string representation of a list of indices where the input list has value 1.
    """
    bool_list = from_str_to_list(bool_str)
    numeric_list = [i for i, val in enumerate(bool_list) if val == 1]
    return str(numeric_list)


# *******************************************************************************
# *************************** MAIN **********************************************
# *******************************************************************************

if __name__ == '__main__':

    # ---------------------------------------------------------------------------
    # Request user input for NREADS, NSAMPLES, NUNITS with default values
    NREADS = get_input("Please set the number of reads", 100) # Number of reads of the chip
    NSAMPLES = get_input("Please set the number of samples", 5) # Number of experiments
    NUNITS = get_input("Please set the number of units", 30)


    start_time = time.time()


    # Z_values = []
    accuracy_EC_values = []
    accuracy_MEC_values = []

    chosen_instances = range(1,11)

    # ---------------------------------------------------------------------------
    for instance in chosen_instances:
        
        print(f"\n\n********* INSTANCE {instance} **********")

        subsets = all_instances[instance]
        U = find_U_from_subsets(subsets)


        u = len(U)
        n = len(subsets)

        solutions = all_solutions[instance]
        print("\nTRUE SOLUTIONS: ", solutions)

        # ------------------------------------------------------------------------
        # Build the Hamiltonian of the problem
        H_A = build_ham_EC(U, subsets)

        # Create a bigger Hamiltonian by copying H_A over num_units subspaces.
        I_chip = np.eye(NUNITS, dtype=int)
        H_A_chip = np.kron(I_chip, H_A)


      
        # ************************************************************************
        # *************************** DWAVE SAMPLING *****************************
        # ************************************************************************

        PROBLEM_NAME = f'EC_instance{instance}_NUNITS{NUNITS}'


        # ------------------------------- QPU -----------------------------------

        print("\nDWAVE SOLUTION (DWaveSampler):")
        from dwave.system import DWaveSampler, EmbeddingComposite
        import dwave.inspector

        sampler = EmbeddingComposite(DWaveSampler(solver=dict(topology__type='zephyr')))

        # # Create a pandas DataFrame and save it to .csv.
        df_tot = []
        for ith_sample in range(1, NSAMPLES+1):
            sampleset = sampler.sample_qubo(H_A_chip, num_reads=NREADS, label=f"sample_{ith_sample}")
            ith_df = sampleset.to_pandas_dataframe() # ith-sample dataset
            df_tot.append(ith_df)
            print(sampleset)
            dwave.inspector.show(sampleset)

        df_tot = pd.concat(df_tot, ignore_index=True)
        # print("df_tot", df_tot)



        # ************************************************************************
        # *************************** POSTPROCESSING *****************************
        # ************************************************************************

        # --------------------   create each unit's dataframe   ------------------
        # -----------------  & count occurrences (for each unit)  ----------------

        # Set how to split df_tot's columns
        num_digits = n * NUNITS # length of a sampled state 
        start_points = np.arange(0, num_digits, n)
        end_points = np.arange(n, num_digits + n, n)

        # Create a dataframe for each unit
        df_all_units = []

        for start, end in zip(start_points, end_points):

            # Copy df_tot's columns from "start" to "end"
            df_unit = df_tot.iloc[:, start:end].copy() 

            # Add a column to represent states as arrays and convert arrays to strings
            df_unit["state"] = df_unit.apply(lambda row: str(row.values), axis=1)

            # Just keep this "state" column
            df_unit = df_unit.loc[:, ['state']]

            # Add the occurrences column
            df_unit["num_occurrences"] = df_tot["num_occurrences"]

            # Append the unit dataframe to the list
            df_all_units.append(df_unit)


        # ---------     Sum counts from each unit dataframe    ---------------

        df_sum = pd.concat(df_all_units).groupby(['state']).sum()

        # Sort in descending order
        df_sum = df_sum.sort_values(['num_occurrences'], ascending=False).reset_index()
        
        # Convert boolean arrays to numeric arrays [0, 1, 1] -> [1, 2]
        df_sum['state']= df_sum['state'].apply(from_bool_to_numeric_str)


        # ---------------     Add labels to states     ------------------------

        # From the solution list find the MEC
        solutions_str = [str(s) for s in solutions]
        MEC = min(solutions_str, key=len, default=None)

        df_sum['label'] = df_sum['state'].apply(lambda x: "MEC" if x==MEC
                                                               else "EC" if x in solutions_str 
                                                               else "feasible" if is_feasible(x, subsets)
                                                               else "xxx")

        # Rearrange columns to move 'label' after 'state'.
        df_sum = df_sum[['state', 'label', 'num_occurrences']]
        print(f"\n******* Sum of the counts of each unit's dataframe *******")

        #-----------------    Print the dataframe with colors   --------------------

        df_print = df_sum.copy()
        
        # Define a function that colors arrays in green if they are a solution.
        color_solutions = lambda x: (colored(x, None, 'on_green') 
                                     if x in solutions_str
                                     else colored(x, 'white', None))

        df_print['state'] = df_print['state'].map(color_solutions)

        # Reset columns or they will be shifted.
        df_print.columns =  [colored('state', 'white', None)] + ["label", "num_occurrences"]

        print(df_print)    

        #---------------------- Save this dataframe as CSV  -------------------------

        # Create a pandas DataFrame and save it to .csv.
        header = f'{PROBLEM_NAME}_NREADS{NREADS}_NSAMPLES{NSAMPLES}'
        csv_path = os.path.join(output_dir, f'{current_datetime}_' + header + '.csv')
        df_sum.to_csv(csv_path, index=False)

        #---------------------- Compute EC and MEC accuracy -------------------------

        # Count the total number of exact covers.
        num_EC = df_sum.loc[df_sum["state"].isin(solutions_str), "num_occurrences"].sum()

        # Count the total number of minimum exact covers.
        num_MEC = df_sum.loc[df_sum['state'].apply(lambda string: string == MEC), 'num_occurrences']
        if not num_MEC.empty:
            num_MEC = num_MEC.iloc[0]
        else:
            num_MEC = 0 

        total = NREADS * NUNITS * NSAMPLES

        accuracy_EC = num_EC/total
        print(f"\nACCURACY_EC = num_EC / total = {num_EC}/{total} = {round(accuracy_EC * 100, 2)}%")

        accuracy_MEC = num_MEC/total
        print(f"ACCURACY_MEC = num_MEC / total = {num_MEC}/{total} = {round(accuracy_MEC * 100, 2)}%")


        # Store results for plotting
        accuracy_EC_values.append(accuracy_EC)
        accuracy_MEC_values.append(accuracy_MEC)

        elapsed_time = time.time() - start_time
        print(f'\nComputation time (s): {elapsed_time}')



    ##################### Plot accuracy and save values for future computations #######################

    from EC_DWave_postprocess import plot_accuracy
    plot_accuracy(accuracy_EC_values, accuracy_MEC_values, chosen_instances,
                  NUNITS, NREADS, NSAMPLES)
    
    # Save plotting data to a JSON file
    with open(os.path.join(output_dir, f'{current_datetime}_accuracy_values.json'), 'w') as file:
        json.dump({'accuracy_EC_values': accuracy_EC_values, 'accuracy_MEC_values': accuracy_MEC_values}, file)
