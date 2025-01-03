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
import json
import os
import time
from typing import List #  for annotations
import sys

import pandas as pd
from termcolor import colored # to color a dataframe

from instances import all_instances, all_solutions
from utils import *


# pylint: disable=bad-whitespace, invalid-name, redefined-outer-name, bad-indentation

def setup_logging(log_filename: str):
    """
    Configures logging to redirect stdout and stderr output to both a file and the console.

    This function replaces `sys.stdout` and `sys.stderr` with a custom `Tee` object that duplicates
    the output to multiple destinations: the console and a specified file.

    Args:
        log_filename (str): The path of the file where logging output will be written.

    Inner Class:
        Tee: A class that writes output to multiple streams (e.g., console and file).

        Attributes:
            files (tuple): A collection of file streams to write the output to.

        Methods:
            write(obj: str): Writes the string `obj` to all registered file streams.
            flush(): Forces a flush on all registered file streams.

    Raises:
        OSError: If the file `log_filename` cannot be opened for writing.

    Example:
        >>> setup_logging("log.txt")
        # Everything printed to stdout and stderr will also be logged in "log.txt".

    Notes:
        - The log file is opened in 'w' mode, overwriting any existing content.
        - It is the caller's responsibility to close the log file if needed, although
          the `Tee` object ensures immediate flushing.
    """
    class Tee(object):
        def __init__(self, *files):
            """
            Initializes the Tee with the specified file streams.

            Args:
                files (tuple): Multiple streams to write output to (e.g., console and file).
            """
            self.files = files

        def write(self, obj: str):
            """
            Writes the string `obj` to all file streams.

            Args:
                obj (str): The string to write.
            """
            for f in self.files:
                f.write(obj)
                f.flush()  # Forces immediate flushing

        def flush(self):
            """
            Forces a flush on all registered file streams.
            """
            for f in self.files:
                f.flush()

    log_file = open(log_filename, 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)


# get current date and time
current_datetime = datetime.now().strftime("%m-%d@%Hh%Mm%Ss")

# Create a directory named with the current datetime
output_dir = f"./{current_datetime}"
os.makedirs(output_dir, exist_ok=True)
print(f"... Saving results in {output_dir} directory ... ")

# Imposta il file di log
# log_file_name = os.path.join(output_dir, f"{current_datetime}_output_log.txt")
# setup_logging(log_file_name)


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

def get_input(prompt, default_value, type_func=int):
    """
    Prompts the user for input with a default value and type conversion.

    Parameters
    ----------
    prompt : str
        The message displayed to the user as the input prompt.
    default_value : any
        The value returned if the user does not provide any input (presses Enter).
    type_func : callable, optional
        A function to convert the user's input to a desired type. Defaults to `int`.

    Returns
    -------
    any
        The user's input converted using `type_func`, or the `default_value` if no input is provided.

    Raises
    ------
    ValueError
        If the user's input cannot be converted using `type_func`.

    Examples
    --------
    >>> get_input("Enter your age", 25)
    Enter your age (default 25): 
    25

    >>> get_input("Enter your age", 25)
    Enter your age (default 25): 30
    30
    """
    user_input = input(f"{prompt} (default {default_value}): ")
    if user_input.strip() == "":
        return default_value
    else:
        return type_func(user_input)


def from_str_to_list(state_as_str: str) -> List[int]:
    """
    Convert a string representation of a list to a list of integers.

    Parameters
    ----------
    state_as_str : str
        A string representation of a list of integers, e.g., "[1, 0, 1]".

    Returns
    -------
    List[int]
        A list of integers extracted from the input string.
    """
    # Remove the square brackets and commas, then split and convert to integers
    return list(map(int, state_as_str.strip("[]").replace(",", "").split()))


def from_bool_to_numeric_lst(bool_list: List[int]) -> List[int]:
    """
    Convert a boolean-like list to a list of indices where the values are 1.

    Parameters
    ----------
    bool_list : List[int]
        A list of integers (0 or 1), e.g., [0, 1, 1].

    Returns
    -------
    List[int]
        A list of indices where the values in the input list are 1.
    """
    return list(np.where(np.array(bool_list) == 1)[0])


def from_bool_to_numeric_str(bool_str: str) -> str:
    """
    Convert a string representation of a boolean-like list to a string of indices.

    Parameters
    ----------
    bool_str : str
        A string representation of a boolean-like list, e.g., "[0,1,1]".

    Returns
    -------
    numeric_str : str
        A string representation of a list of integers, e.g., "[2, 3]".
    """
    bool_list = from_str_to_list(bool_str)
    numeric_list = from_bool_to_numeric_lst(bool_list)
    numeric_str = str(numeric_list)
    return numeric_str


# ************************************************************************
# *************************** MAIN ***************************************
# ************************************************************************

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

    chosen_instances = [3]

    # ---------------------------------------------------------------------------
    for instance in chosen_instances:
        
        print(f"\n\n********* INSTANCE {instance} **********")

        subsets = all_instances[instance]
        U = find_U_from_subsets(subsets)


        u = len(U)
        n = len(subsets)

        solutions = all_solutions[instance]
        print("\nTRUE SOLUTIONS: ", solutions)

        # ---------------------------------------------------------------------------
        # Build the Hamiltonian of the problem
        H_A = build_ham(U, subsets)

        # Create a bigger Hamiltonian by copying H_A over num_units subspaces.
        I_chip = np.eye(NUNITS, dtype=int)
        H_A_chip = np.kron(I_chip, H_A)

      
        # ************************************************************************
        # *************************** DWAVE SAMPLING *****************************
        # ************************************************************************

        PROBLEM_NAME = f'EC_instance{instance}_NUNITS{NUNITS}'
        

        # ------------------------- SIMULATED ANNEALING ------------------------ 

        # print("\nDWAVE SOLUTION (SimulatedAnnealingSampler):")
        # import neal
        # solver = neal.SimulatedAnnealingSampler()
        # NREADS = 3
        # sampleset = solver.sample_qubo(H_A_chip, num_reads=NREADS)
        
        # df = sampleset.to_pandas_dataframe()
        # csv_path = f"./{PROBLEM_NAME}_{NREADS}_{current_datetime}.csv"
        # df.to_csv(csv_path, index=False)
        # print(sampleset)

        # --------------------------- QPU ---------------------------------------

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
            # print(sampleset)
            # dwave.inspector.show(sampleset)

        df_tot = pd.concat(df_tot, ignore_index=True)
        # print("df_tot", df_tot)

        # ************************************************************************
        # *************************** POSTPROCESSING *****************************
        # ************************************************************************

        # --------------------   create each unit's dataframe   --------------------
        # -----------------  & count occurrences (for each unit)  ------------------

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


        # ---------      Sum counts from each unit dataframe    --------------------

        df_sum = pd.concat(df_all_units).groupby(['state']).sum()

        # Sort in descending order
        df_sum = df_sum.sort_values(['num_occurrences'], ascending=False).reset_index()
        
        # Convert boolean arrays to numeric arrays [0, 1, 1] -> [1, 2]
        df_sum['state']= df_sum['state'].apply(from_bool_to_numeric_str)


        #####################################
        solutions_str = [str(s) for s in solutions]

        # Trova MEC (assicurati che solutions non sia vuoto)
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

        #-----------------------------------------------------------------------------

        # Create a pandas DataFrame and save it to .csv.
        header = f'{PROBLEM_NAME}_NREADS{NREADS}_NSAMPLES{NSAMPLES}'
        csv_path = os.path.join(output_dir, f'{current_datetime}_' + header + '.csv')
        df_sum.to_csv(csv_path, index=False)

        #-----------------------------------------------------------------------------

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


    # Salvataggio in un file JSON dei dati da plottare
    with open(os.path.join(output_dir, f'{current_datetime}_accuracy_values.json'), 'w') as file:
        json.dump({'accuracy_EC_values': accuracy_EC_values, 'accuracy_MEC_values': accuracy_MEC_values}, file)

    ###################################################################################
    #################################  PLOT   #########################################
    ###################################################################################

    from EC_DWave_postprocess import plot_accuracy
    plot_accuracy(accuracy_EC_values, accuracy_MEC_values, chosen_instances,
                  NUNITS, NREADS, NSAMPLES)

