import time
import pprint
from typing import List

import numpy as np
import pandas as pd
import ast # to evaluate strings as arrays
from termcolor import colored # to color a dataframe

from utils import get_counts, from_str_to_arr, is_feasible


if __name__ == '__main__':
    start_time = time.time()

    # Instance 3 of the e-mail (dimension=6)
    subsets_as_list = [{8, 10, 3}, {2, 4, 5, 6, 9, 11, 12}, {1, 3, 7, 8, 10}, 
                       {3, 4, 6, 8, 11}, {2, 3, 4, 6, 7, 9, 12}, {1, 7}]

    s = len(subsets_as_list) # dimension of the problem
    num_units = 10 # number of parallel works
                   # we will create one dataframe for each unit
    NREADS = 100


    num_digits = s*num_units # length of a sample 

    SOLUTIONS = [[0,1,5], [1,2]]
    print("solutions: ", SOLUTIONS)

    # --------------------------------------------------------------------------
    # --------------------   display the global dataframe   --------------------

    # df_tot is the dataframe containing the whole information of one run.
    csv_path = f"./EC_instance3_u12_s{s}_{num_units}units_{NREADS}_Adv2_1of5.csv"


    df_tot = pd.read_csv(csv_path)

    # Add a column to represent states as arrays.
    df_tot["array"] = list(df_tot.iloc[:, 0:num_digits].to_numpy(dtype=int))

    # Drop samples' columns (all the info is now in "array" column).
    # df_tot.drop(df_tot.iloc[:, 0:num_digits], inplace=True, axis=1)

    # Turn the arrays to strings (arrays are not hashable).
    df_tot["array"] =  df_tot["array"].map(str)

    print("\n--df_tot:\n", df_tot.head(10))


    # --------------------        count occurrences         --------------------
    
    df_counts = get_counts(df_tot, col="array", print_df_counts=True)



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

        #print(f"\n******* Unit number {i} *******", df_unit.head())

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


    # -----------------        Print it with colors         --------------------

    df_print = df_sum.copy()

    df_print['is_feasible'] = df_print['array_to_num'].apply(lambda x: is_feasible(x, subsets_as_list))

    # Define a function that colors arrays in green if they are a solution.
    color_solutions = lambda x: (colored(str(x), None, 'on_green') 
                                 if eval(x) in SOLUTIONS 
                                 else colored(x, 'white', None))

    df_print['array_to_num'] = df_print['array_to_num'].map(color_solutions)

    # Reset columns or they will be shifted.
    df_print.columns =  [colored('array_to_num', 'white', None)] + ["counts", "is_feasible"]
    print(df_print.head(10))

    # --------------------------------------------------------------------------

    elapsed_time = time.time() - start_time
    print(f'\nComputation time (s): {elapsed_time}')
