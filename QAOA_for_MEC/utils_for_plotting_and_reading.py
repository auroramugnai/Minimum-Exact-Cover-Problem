from __future__ import annotations

import csv
from datetime import datetime
import math
import os
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from more_itertools import distinct_permutations
import seaborn as sns

from utils_to_study_an_instance import define_instance, find_spectrum


def highlight_correct_ticks(ax: Axes, EXACT_COVERS: List[str]) -> None:
    """
    Highlights ticks on the axis corresponding to exact covers, minimum exact covers (MEC), 
    and specific binary state permutations.

    - If the state is an exact cover, the tick color is red (crimson).
    - If it is a minimum exact cover (MEC), the tick color is green (limegreen).
    - If it matches a specific "1" permutation, the tick color is light grey.
    - Otherwise, the tick color is black.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object whose ticks will be modified.
    EXACT_COVERS : list of str
        A list of exact cover states, where each state is a binary string.

    Returns
    -------
    None
        The function modifies the tick labels of the provided axis directly.
    """
    
    # Retrieve the current tick labels
    xlabels = [elem.get_text() for elem in ax.xaxis.get_ticklabels()]
    
    # Determine the minimum number of 1s in the exact covers to identify MECs
    minimum_length = min([x.count("1") for x in EXACT_COVERS])
    
    # Find the Minimum Exact Cover (MEC) - the first state with the minimum number of 1s
    MEC = [state for state in EXACT_COVERS if state.count("1") == minimum_length][0]
    
    # Get the length of the MEC to generate distinct "1" permutations
    n = len(MEC)
    
    # Generate all possible distinct permutations of a string with n-1 zeros and one 1
    one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
    
    # Iterate through each tick label and modify the color based on the criteria
    for (state, ticklbl) in zip(xlabels, ax.xaxis.get_ticklabels()):
        if state == MEC:
            # If the state matches MEC, set tick color to limegreen
            ticklbl.set_color('limegreen')
        elif state in EXACT_COVERS:
            # If the state is an exact cover, set tick color to crimson (red)
            ticklbl.set_color('crimson')
        elif state in one_one_states:
            # If the state matches a specific "1" permutation, set tick color to grey
            ticklbl.set_color('grey')
        else:
            # Otherwise, set tick color to black
            ticklbl.set_color('black')



#############################################################################################################
#############################################################################################################
#############################################################################################################

def underline_states(ax: Axes, states_to_underline: List[str], fontsize: int) -> None:
    """
    Underlines specific states on the x-axis of a plot.

    This function iterates through the x-axis tick labels and underlines
    those states (given as binary strings) that are specified in the
    `states_to_underline` list. The underlining is done by annotating
    underscores beneath the tick labels, with customizable font size.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object whose states will be underlined.
    states_to_underline : list of str
        A list of states (as strings) to underline. Each state is expected to be a binary string.
    fontsize : int
        Font size for the underlines.

    Returns
    -------
    None
        The function modifies the plot directly by adding annotations for the underlines.
    """
    
    # Iterate through each tick label on the x-axis
    for elem in ax.xaxis.get_ticklabels():
        # If the tick label text matches one of the states to underline
        if elem.get_text() in states_to_underline:
            # Calculate the length of the state string to adjust the underline length
            l = len(elem.get_text())
            
            # Annotate the plot with underscores as the underline
            ax.annotate('_' * l,  # Create a string of underscores equal to the length of the state
                        xy=(elem.get_position()[0], 0),  # Position the annotation at the tick's x-position
                        xytext=(0, -10),  # Shift the annotation 10 units below the tick
                        xycoords=('data', 'axes fraction'),  # Use axis coordinates for placement
                        textcoords='offset points',  # Use offset points for text positioning
                        ha='center', va='top',  # Align the annotation in the center horizontally and top vertically
                        rotation=90,  # Rotate the underline to make it horizontal
                        fontsize=fontsize,  # Set the font size of the underline
                        color='grey')  # Set the underline color to grey



#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_histogram_of_df_column(df: pd.DataFrame, 
                                column_to_plot: str, 
                                EXACT_COVERS: List[str], 
                                states_to_underline: List[str], 
                                title: str = '') -> Axes:
    """
    Plots a histogram of a specified column from a dataframe, highlighting exact covers and underlining specific states.

    This function creates a bar plot of the specified column in the dataframe, showing the percentage distribution 
    of each state in the column. Exact cover states are highlighted, and specific states are underlined on the x-axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a 'states' index and numerical data.
    column_to_plot : str
        The column name to plot as a histogram.
    EXACT_COVERS : list of str
        A list of exact cover states to highlight. Each state is a binary string.
    states_to_underline : list of str
        A list of states to underline on the x-axis. Each state is a binary string.
    title : str, optional
        The title of the histogram plot (default is '').

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object for the plot, which can be used for further modifications or display.
    """
    
    # Set the 'states' column as the index of the dataframe
    df = df.set_index('states')
    
    # Ensure the data is numeric and handle missing values by filling with 0.0
    df = df.astype(float).fillna(0.0)
    
    # ##### COMPUTE PERCENTAGES
    total = df.sum()  # Sum of values in the dataframe
    percentage = (df / total) * 100  # Convert values to percentages

    # Extract the percentage of the specified column and sort it
    percentage = percentage[[column_to_plot]]
    percentage = percentage.sort_values(column_to_plot, ascending=False)
    
    ##### PLOT FIGURE
    plt.figure(figsize=(10, 5))
    N = 10  # Font size for the labels
    ax = sns.barplot(x="states", y=column_to_plot, data=percentage, 
                     width=0.7, color='red', alpha=0.5)
    
    # Add percentage labels on top of the bars
    labels = percentage[column_to_plot].round(1).astype('str') + '%'
    for container in ax.containers:
        ax.bar_label(container, labels=labels, fontsize=N-3)
    
    ### Highlight exact covers' ticks and underline initial states
    df_for_ticks = percentage.copy()
    df_for_ticks["states"] = df_for_ticks.index 
    underline_states(plt.gca(), states_to_underline, fontsize=N+2)  # Underline the specific states
    highlight_correct_ticks(plt.gca(), EXACT_COVERS)  # Highlight exact covers
    
    ### Refine plot aesthetics
    plt.xlabel("States", fontsize=N)
    plt.ylabel("", fontsize=N)
    plt.xticks(fontsize=N-2, rotation="vertical")  # Rotate x-axis labels
    plt.yticks(fontsize=N)
    plt.xlim(xmin=-1)  
    plt.ylim(ymin=0, ymax=106) 
    plt.minorticks_on()  
    plt.grid(alpha=0.2)  
    plt.title(title, fontsize=N)  

    return ax



#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_histogram_of_best_column(df: pd.DataFrame, 
                                  best_column: str, 
                                  EXACT_COVERS: List[str], 
                                  states_to_underline: List[str], 
                                  title: str = '') -> None:
    """
    Plots the histogram of the best column from a dataframe, highlighting exact covers and underlining specific states.
    Also overlays error bars for the average values.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a 'states' index and numerical data.
    best_column : str
        The column name considered the best and used for histogram values.
    EXACT_COVERS : list of str
        A list of exact cover states to highlight.
    states_to_underline : list of str
        A list of states to underline on the x-axis.
    title : str, optional
        The title of the histogram plot (default is '').

    Returns
    -------
    None
    """
    # Generate the basic histogram for the best column
    ax = plot_histogram_of_df_column(df, best_column, EXACT_COVERS, states_to_underline, title=title)
    
    # Set the dataframe index to 'states' and ensure numeric conversion
    df = df.set_index('states').astype(float).fillna(0.0)

    # Compute percentages and add average and standard deviation columns
    total = df.sum()
    percentage = (df / total) * 100
    
    # Calculate the average and standard deviation for each state
    percentage['average'] = percentage.mean(numeric_only=True, axis=1)
    percentage['std'] = percentage[percentage.columns[:-1]].std(numeric_only=True, axis=1)
        
    # Keep only the best column, average, and standard deviation results
    percentage = percentage[[best_column, "average", "std"]]
    percentage = percentage.sort_values(best_column, ascending=False)
    
    # Overlay error bars for the average values
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]  # Get the x positions of bars
    y_coords = percentage["average"]  # Average values for the y-axis
    ax.errorbar(x=x_coords, y=y_coords, yerr=percentage["std"], linestyle="",
                markerfacecolor='none', linewidth=1,
                marker='o', color='k', ecolor='k', elinewidth=0.7, capsize=3.5, 
                barsabove=True, alpha=0.5)  # Plot error bars for averages
    plt.show()



#############################################################################################################
#############################################################################################################
#############################################################################################################

def define_parameters_from_filename(filename: str, verbose: bool) -> tuple:
    """
    Extracts parameters from a filename string.

    Parameters
    ----------
    filename : str
        The filename containing parameter details, such as "dim6_mail5_all0_random_p3_10ra_k0.085_".
    verbose : bool
        If True, prints the extracted parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - n (int): Instance dimension.
        - instance (int): Instance number.
        - init_name (str): Initialization type.
        - p (int): Maximum layer or circuit depth.
        - ra (int): Number of random attempts.
        - k (float): Problem parameter (e.g., `k = l1/l2 * n`).

    Example
    -------
    >>> define_parameters_from_filename("dim6_mail3_all0_random_p3_10ra_k0.067_", verbose=True)
    n=6, i=3, init=all0, p=3, ra=10, k=0.067
    """
    # Remove path information if present and isolate the filename
    filename = filename.rsplit('/', 1)[1]

    # Extract the instance dimension (n)
    n = int(filename.split('dim')[1][0])

    # Extract the instance number
    instance = filename.split('mail')[1]
    instance = int(instance.split('_')[0])

    # Extract the initialization type (e.g., "all0" or "customized")
    if "all" in filename:
        init_name = filename.split(f'mail{instance}_')[1]
        init_name = init_name.split("_")[0]
    else:
        init_name = 'customized'

    # Extract the maximum layer or circuit depth (p)
    p = int(filename.split('_p')[1].split('_')[0])

    # Extract the number of random attempts (ra)
    ra = filename.split(f'p{p}_')[1]
    ra = int(ra.split("ra")[0])

    # Extract the problem parameter (k)
    k = filename.split("ra_k")[1]
    k = float(k.split("_")[0])

    # Print extracted parameters if verbose is True
    if verbose:
        print(f"n={n}, i={instance}, init={init_name}, p={p}, ra={ra}, k={k}")

    # Return the extracted parameters as a tuple
    return n, instance, init_name, p, ra, k



#############################################################################################################
#############################################################################################################
#############################################################################################################

# Function to extract the instance number from the filename
def extract_instance_from_filename(filename: str) -> int:
    """
    Extracts the instance number from the filename using the `define_parameters_from_filename` function.
    
    Parameters
    ----------
    filename : str
        The filename to extract the instance number from.
    
    Returns
    -------
    int
        The extracted instance number.
    """
    _, instance, _, _, _, _ = define_parameters_from_filename(filename, verbose=False)
    return instance



#############################################################################################################
#############################################################################################################
#############################################################################################################

def find_files_containing_string(strings: list, path: str, verbose: bool = False) -> tuple:
    """
    Finds files that contain specific substrings in their names and sorts them based on an extracted parameter.

    Parameters
    ----------
    strings : list of str
        A list of substrings to search for in the file names.
    path : str
        The directory path where the files should be searched.
    verbose : bool, optional
        If True, prints the found files and their associated data files (default is False).

    Returns
    -------
    tuple
        A tuple containing:
        - FILENAME_list : list of str
            List of files that match the substring criteria, excluding '_data' in the filename.
        - DATA_FILENAME_list : list of str
            List of files that match the substring criteria and contain '_data' in the filename.
    
    Example
    -------
    >>> find_files_containing_string(['dim6', 'mail3'], './data/', verbose=True)
    """
    # Initialize empty lists to store normal files and data files
    FILENAME_list = []
    DATA_FILENAME_list = []

    # Loop over all files in the specified directory
    for obj in os.listdir(path):
        file_path = os.path.join(path, obj)
        
        # Check if the object is a file and contains all substrings in `strings`
        if os.path.isfile(file_path) and np.all([s in obj for s in strings]):
            if '_data' in obj:
                DATA_FILENAME_list.append(file_path)
            else:
                FILENAME_list.append(file_path)

    # Sort both lists based on the extracted instance number
    FILENAME_list.sort(key=lambda x: extract_instance_from_filename(x))
    DATA_FILENAME_list.sort(key=lambda x: extract_instance_from_filename(x))

    # Print file paths if verbose is True
    if verbose:
        for f, d in zip(FILENAME_list, DATA_FILENAME_list):
            print(f"\n{f}\n{d}\n\n")

    # Return the sorted lists
    return FILENAME_list, DATA_FILENAME_list



#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_file(FILENAME: str, DATA_FILENAME: str, colorchosen: str, alpha: float,
              states_to_underline: Optional[List[str]] = None, 
              title: Optional[str] = None,
              dont_show_in_title: List[str] = [], 
              pars: List[Tuple] = [], figsize: Tuple[int, int] = (18, 8), 
              dpi: int = 300, N: int = 10) -> None:
    """
    Plots a bar chart for the states in a given file, highlighting exact covers, underlining initial states, 
    and adding error bars for the average values. It customizes the appearance and provides an informative title.

    Parameters
    ----------
    FILENAME : str
        Path to the CSV file containing the state data.
    DATA_FILENAME : str
        Path to the data file with metadata and parameter information.
    colorchosen : str
        Color for the bars in the plot.
    alpha : float
        Transparency level for the bars.
    states_to_underline : list of str, optional
        List of states to underline on the x-axis (default is None).
    title : str, optional
        Title of the plot (default is None).
    dont_show_in_title : list of str, optional
        List of parameters not to show in the plot's title (default is empty list).
    pars : list of tuples, optional
        Parameters to override the extraction from the filename (default is empty list).
    figsize : tuple of int, optional
        Size of the figure (default is (18, 8)).
    dpi : int, optional
        DPI for the plot (default is 300).
    N : int, optional
        Font size for the plot elements (default is 10).

    Returns
    -------
    None
        This function displays the plot but does not return anything.
    
    Example
    -------
    >>> plot_file("data.csv", "metadata.txt", colorchosen="blue", alpha=0.7)
    """
    
    # Initialize the figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    # Extract parameters from filename if not provided
    if not pars:
        n, instance, init_name, p, random_attempts, k = define_parameters_from_filename(DATA_FILENAME, verbose=False)
    else:
        n, instance, init_name, p, random_attempts, k = pars

    print(n, instance, init_name, p, random_attempts, k)
    # Define the problem instance using the extracted parameters
    U, subsets_dict = define_instance(n, instance, verbose=False)

    print(subsets_dict)
    # Analyze spectrum to extract relevant state information
    states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k=1)
    print("EXACT COVERS", EXACT_COVERS)
    MEC = min(EXACT_COVERS, key=lambda s: s.count('1'))  # Minimum exact cover

    # Load the state data into a pandas DataFrame
    df = pd.read_csv(FILENAME, dtype=str).set_index('states')
    df = df.astype(float).fillna(0.0)

    # Compute percentages and add columns for average and standard deviation
    total = df.sum()
    percentage = (df / total) * 100
    percentage['average'] = percentage.mean(numeric_only=True, axis=1)
    percentage['std'] = percentage[percentage.columns[:-1]].std(numeric_only=True, axis=1)

    # Identify the best histogram index based on metadata file
    with open(DATA_FILENAME, 'r') as DATA_FILE:
        for line in DATA_FILE:
            if 'Attempt that reached the best result with' in line:
                string = line.split('#')[1]
                i_best = string.split(' ')[0]

    # Select the column corresponding to the best result
    column_best = f'counts_p{p}_{i_best}of{random_attempts}'

    # Keep the best and average results for plotting
    percentage = percentage[[column_best, "average", "std"]]
    percentage = percentage.sort_values(column_best, ascending=False)

    ############################# LABELS ###################################
    # Create subplot for bar chart
    ax = sns.barplot(x="states", y=column_best, data=percentage, width=0.7, color=colorchosen, alpha=alpha)
    
    # Add edge color to the bars for better visibility
    for bar in ax.patches:
        bar.set_edgecolor('indigo')

    # Annotate bars with percentage values
    labels_df = percentage[column_best].round(1).astype(str).add('%')
    labels_df = labels_df.where(labels_df.index.isin(EXACT_COVERS), "")

    # Annotate manually to adjust x and y positions
    for container in ax.containers:
        for rect, label in zip(container, labels_df.tolist()):
            label_x = rect.get_x()
            label_y = rect.get_height()
            ax.text(label_x, label_y, label, fontsize=N, ha='left', va='bottom')

    # Add error bars for average values
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = percentage["average"]
    ax.errorbar(x=x_coords, y=y_coords, yerr=percentage["std"], linestyle="",
                markerfacecolor='none', linewidth=1, marker='o', color='k', ecolor='k', 
                elinewidth=0.7, capsize=3.5, barsabove=True, alpha=0.6)

    ########################### HIGHLIGHT #####################################
    # Highlight exact covers
    df_for_ticks = percentage.copy()
    df_for_ticks["states"] = df_for_ticks.index
    highlight_correct_ticks(ax, EXACT_COVERS)

    # Underline the initial states based on initialization type
    if init_name == "all1":
        one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
        init_string = "$|1 \\rangle$-initialization"
        underline_states(ax, one_one_states, fontsize=N+2)
    elif init_name == "all0":
        underline_states(ax, "0" * n, fontsize=N+2)
        init_string = "$|0 \\rangle$-initialization"
    else:
        underline_states(ax, states_to_underline, fontsize=N+4)
        init_string = ""

    ############################### TITLE #################################
    # Construct subplot title with the provided parameters
    dictstring = {"n": f"$n={n}$", "i": f"Instance #{instance}", "init": f"{init_string}", 
                  "p": f"$p={p}$", "ra": f"$ra={random_attempts}$", "k": f"$k={k}$"}
    title_string = ', '.join([dictstring[x] for x in dictstring if x not in dont_show_in_title])

    # Append bounds and parameters information to the title
    bounds_and_pars0 = FILENAME.split('pars0')[1].split('.csv')[0]
    bounds_and_pars0 = bounds_and_pars0.replace("pi", "\pi").replace("x", "\\times")
    title_string += f"\n${bounds_and_pars0}$"

    ax.set_title(title_string, fontsize=N)

    ################################################################
    # Refine plot aesthetics
    plt.xlabel("states", fontsize=N)
    plt.ylabel("percentage [%]", fontsize=N)
    plt.xticks(fontsize=N, rotation="vertical")
    plt.yticks(fontsize=N)
    plt.xlim(xmin=-1)
    plt.ylim(ymin=0, ymax=106)
    plt.minorticks_on()
    plt.grid(alpha=0.2)

    # Adjust subplot layout and display the figure
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.show()



#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_list_of_files(FILENAME_list: List[str], DATA_FILENAME_list: List[str], colorchosen: str, alpha: float,
                       init_name: Optional[str] = None,
                       states_to_underline: Optional[List[str]] = None, 
                       title: Optional[str] = None,
                       dont_show_in_title: List[str] = [], 
                       dont_show_in_titles: List[str] = [], 
                       pars: List[Optional[int]] = [], figsize: tuple = (8, 8), dpi: int = 100, N: int = 10):
    """
    Plots multiple subplots for each file in the FILENAME_list. Each subplot visualizes the percentage of states,
    average values, and error bars from the corresponding data file.

    Parameters
    ----------
    FILENAME_list : list of str
        List of file paths to CSV files containing state data for plotting.
    DATA_FILENAME_list : list of str
        List of file paths to metadata files for each corresponding CSV file.
    colorchosen : str
        Color to be used for the bars in the plot.
    alpha : float
        Transparency level of the bars in the plot.
    init_name : str, optional
        Initialization type (e.g., "all0" or "all1"). If None, no initialization is displayed.
    states_to_underline : list of str, optional
        States to be underlined in the plot.
    title : str, optional
        Title of the entire figure. If None, a title will be generated from the parameters.
    dont_show_in_title : list of str, optional
        List of elements to exclude from the figure's title.
    dont_show_in_titles : list of str, optional
        List of elements to exclude from individual subplot titles.
    pars : list of optional int, optional
        A list of parameters (n, instance, p, etc.) to override the filename-based extraction.
    figsize : tuple of int, default (18, 8)
        Size of the entire figure.
    dpi : int, default 300
        Resolution of the figure.
    N : int, default 10
        Font size for labels and titles.

    Returns
    -------
    None
        The function displays the plot and saves it as a PDF file.
    """
    
    # Initialize the figure.
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Number of rows and columns for subplots (fixed to 5x2 grid).
    num_rows = math.ceil(len(FILENAME_list)/2)
    num_cols = 2
    
    # Iterate over pairs of FILENAME and DATA_FILENAME.
    for num_subplot, (FILENAME, DATA_FILENAME) in enumerate(zip(FILENAME_list, DATA_FILENAME_list)):
        # Extract parameters from filename if not provided.
        if not pars:
            n, instance, init_name, p, random_attempts, k = define_parameters_from_filename(DATA_FILENAME, verbose=False)
        else:
            n, instance, init_name, p, random_attempts, k = pars

        # Define the problem instance based on extracted parameters.
        U, subsets_dict = define_instance(n, instance, verbose=False)

        print("in the function subsets_dict is ", subsets_dict)

        # Analyze spectrum to extract relevant state information.
        states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k=1)

        MEC = min(EXACT_COVERS, key=lambda s: s.count('1'))  # MEC is the minimal energy cover

        # Load CSV data into pandas DataFrame and process it.
        df = pd.read_csv(FILENAME, dtype=str).set_index('states')
        df = df.astype(float).fillna(0.0)

        # Compute percentage of each state, along with the average and standard deviation.
        total = df.sum()
        percentage = (df / total) * 100
        percentage['average'] = percentage.mean(numeric_only=True, axis=1)
        percentage['std'] = percentage[percentage.columns[:-1]].std(numeric_only=True, axis=1)

        # Read metadata to extract the attempt that reached the best result.
        with open(DATA_FILENAME, 'r') as DATA_FILE:
            for line in DATA_FILE:
                if 'Attempt that reached the best result with' in line:
                    string = line.split('#')[1]
                    i_best = string.split(' ')[0]  # Best attempt index

        # Construct the column name for the best result.
        column_best = f'counts_p{p}_{i_best}of{random_attempts}'

        # Select best and average results for plotting.
        percentage = percentage[[column_best, "average", "std"]]
        percentage = percentage.sort_values(column_best, ascending=False)

        ############################# LABELS ###################################
        # Create a subplot for this particular file.
        fig.add_subplot(num_rows, num_cols, num_subplot + 1)
        ax = sns.barplot(x="states", y=column_best, data=percentage, width=0.7, 
                         facecolor=colorchosen, edgecolor=colorchosen, alpha=alpha)

        # Annotate bars with percentage values.
        labels_df = percentage[column_best].round(1).astype(str).add('%')
        labels_df = labels_df.where(labels_df.index.isin([MEC]), "")  # Annotate only MEC

        # Manually position the percentage labels on the bars.
        for container in ax.containers:
            for rect, label in zip(container, labels_df.tolist()):
                label_x = rect.get_x()
                label_y = rect.get_height()
                ax.text(label_x, label_y, label, fontsize=N-2, ha='left', va='bottom')
                
        # Add error bars for average values.
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = percentage["average"]
        ax.errorbar(x=x_coords, y=y_coords, yerr=percentage["std"], linestyle="",
                    markerfacecolor='none', linewidth=1, marker='o', color="k", ecolor="k", 
                    elinewidth=0.7, capsize=3.5, barsabove=True, alpha=1)

        ########################### HIGHLIGHT #####################################
        # Highlight exact covers by adjusting ticks.
        df_for_ticks = percentage.copy()
        df_for_ticks["states"] = df_for_ticks.index
        highlight_correct_ticks(ax, EXACT_COVERS)

        # Sottolinea gli stati iniziali (underline initial states based on init_name)
        if init_name == "all1":
            one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
            init_string = "$|1 \\rangle$-initialization"
        elif init_name == "all0":
            init_string = "$|0 \\rangle$-initialization"
        else:
            init_string = ""

        ############################### TITLE #################################
        # Construct the title for the subplot.
        dictstring = {
            "n": f"$n={n}$", "i": f"Instance #{instance}", "init": f"{init_string}", 
            "p": f"$p={p}$", "ra": f"$ra={random_attempts}$", "k": f"$k={k}$"
        }
        title_string = ', '.join([dictstring[x] for x in dictstring if x not in dont_show_in_titles])  # Subplot title
        figure_title_string = ', '.join([dictstring[x] for x in dictstring if x not in dont_show_in_title])  # Figure title
        
        # Add bounds information to the title.
        bounds_and_pars0 = FILENAME.split('pars0')[1].split('.csv')[0]
        bounds_and_pars0 = bounds_and_pars0.replace("pi", "\pi").replace("x", "\\times")
        title_string += f"\n${bounds_and_pars0}$"

        ax.set_title(title_string, fontsize=N)
        
        ################################################################
        if num_subplot in [0, 4, 8]:  # Add ylabel only for certain subplots
            ax.set_ylabel("percentage [%]", fontsize=N)
        else:
            ax.set_ylabel("")

        # Refine plot aesthetics: set labels and grid.
        plt.xlabel("states", fontsize=N)
        plt.xticks(fontsize=N-3, rotation="vertical")
        plt.yticks(fontsize=N)
        plt.xlim(xmin=-1)
        plt.ylim(ymin=0, ymax=110)
        plt.minorticks_on()
        plt.grid(alpha=0.2)

    # Save the figure with a timestamp.
    current_datetime = datetime.now().strftime("@%Y-%m-%d@%Hh%Mm%Ss")
    plt.savefig(f"all1_random_{current_datetime}.pdf")
    
    # Display the plot.
    plt.tight_layout()
    plt.show()



#############################################################################################################
#############################################################################################################
#############################################################################################################

def find_files_containing_string_parameters_fixing(directory: str, substring: str) -> Tuple[List[str], List[List[str]]]:
    """
    Searches for files in a given directory that match a certain substring, filters them to find
    those related to data, and then groups them based on a common prefix. It returns a sorted list of 
    data files and a list of lists of other files grouped by their associated data file.

    Parameters
    ----------
    directory : str
        The path of the directory to search in.
    substring : str
        A string that must appear in the filenames to be considered.

    Returns
    -------
    datafiles : list of str
        A list of filenames that contain '_data.txt' and match the given substring.
    list_of_list_of_files : list of list of str
        A list of lists, where each sublist contains filenames that match a given data file, 
        grouped by a common prefix.
    """
    
    # Get all files in the specified directory.
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Filter files to include only those containing the given substring.
    files = [f for f in files if substring in f]
    
    # Initialize a list to store data files (those ending with '_data.txt').
    datafiles = [f for f in files if f.endswith("_data.txt")]
    
    # Sort the data files based on the instance number extracted from their names.
    datafiles.sort(key=lambda x: extract_instance_from_filename(x))

    # Initialize an empty list to store lists of other related files.
    list_of_list_of_files = []
    
    # For each data file, find other files that have the same prefix (before '_p').
    for d in datafiles:
        header = d.split("_data")[0]  # Extract prefix before '_data'
        
        list_of_files = []
        for f in files:
            # If the file isn't a data file and has the same prefix, add it to the list.
            if not f in datafiles and f.rsplit("_p", 1)[0] == header:
                list_of_files.append(f)
        
        # Append the list of related files for the current data file.
        list_of_list_of_files.append(list_of_files)
    
    return datafiles, list_of_list_of_files



#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_file_parameter_fixing(path: str, datafile: str, associated_files: List[str], init_name: str, 
                                title: str = None, dont_show_in_title: List[str] = [], 
                                figsize: tuple = (16, 10), N: int = 10, dpi: int = 300):
    """
    Plots a bar chart of percentages for different states based on given data files, highlighting
    certain states and marking initialization states. The function processes associated files, 
    normalizes the data, and visualizes the results in a stacked bar chart.

    Parameters
    ----------
    path : str
        The directory path where the data files are located.
    datafile : str
        The main data file containing the state information.
    associated_files : list of str
        A list of associated files that contain data to merge with the main data file.
    init_name : str
        The type of initial state (e.g., "all1", "all0") to be highlighted in the plot.
    title : str, optional
        The title of the plot. If not provided, it is generated from the data.
    dont_show_in_title : list of str, optional
        A list of parameters to exclude from the title string.
    figsize : tuple, optional
        The size of the figure, default is (16, 10).
    N : int, optional
        Font size for labels, default is 10.
    dpi : int, optional
        The resolution of the plot, default is 300.

    Returns
    -------
    None
        Displays the plot directly.
    """
    
    # Create the figure and axis for plotting.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    ################################ EXTRACT PARAMETERS ################################
    # Extract parameters from the datafile name.
    n, instance, init_name, maxp, random_attempts, k = define_parameters_from_filename(path + datafile, verbose=False)
    plot_title = f"n={n}, i={instance}, init={init_name}, maxp={maxp}, ra={random_attempts}, k={k}"

    # Define the problem instance.
    U, subsets_dict = define_instance(n, instance, verbose=False)
    states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k)
    
    ##################################################################################
    df_final = None

    # Iterate through each associated file and merge them.
    for file_to_read in associated_files:
        df = pd.read_csv(path + file_to_read, dtype={'states': "str"})

        # Merge the dataframes if not the first iteration.
        if df_final is None:
            df_final = df
        else:
            df_final = pd.merge(df_final, df, on="states", how="outer")

    # Fill NaN values with 0 and normalize the counts.
    df_final = df_final.fillna(0)
    for p in range(1, maxp + 1):
        column_name = f'counts_p{p}'
        total = df_final[column_name].sum()
        if total > 0:
            df_final[column_name] = (df_final[column_name] / total) * 100

    # Sort the columns based on "p" to maintain numerical order.
    sorted_columns = sorted(
        [col for col in df_final.columns if col.startswith('counts_p')],
        key=lambda x: int(x.split('p')[1])
    )

    ################################ PLOT ################################
    # Define colors for the bars.
    colors = ["darkorange", "crimson", "indigo"]

    # Plot the data as a bar chart.
    df_final.plot(
        x='states', kind="bar", width=0.7, fontsize=N, stacked=False, ax=ax, legend=True,
        color=colors, alpha=1
    )

    ################################ Add labels to the bars ###############################
    # Annotate the bars with percentage values.
    for state in df_final['states']:
        max_height = 0
        label_text = ""
        x_position = None

        # Find the maximum height for each state and label it.
        for i, container in enumerate(ax.containers):
            p = i + 1
            row = df_final[df_final['states'] == state]
            if not row.empty:
                value = row[f'counts_p{p}'].values[0]
                if value > max_height:
                    max_height = value
                    label_text = f"{value:.1f}%"
                    x_position = container[df_final[df_final['states'] == state].index[0]].get_x() + container[0].get_width() / 2

        # Add text label on the bars if max height is positive.
        if max_height > 0:
            ax.text(
                x_position, max_height,
                label_text, ha='center', va='bottom', fontsize=N
            )

    ############################### HIGHLIGHT AND UNDERLINE STATES ###############################
    # Highlight the exact covers (correct ticks).
    highlight_correct_ticks(ax, EXACT_COVERS)

    # Underline initial states based on the initialization type.
    if init_name == "all1":
        one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
        init_string = "$|1 \\rangle$-initialization"
        underline_states(ax, one_one_states, fontsize=N + 2)
    elif init_name == "all0":
        underline_states(ax, "0" * n, fontsize=N + 2)
        init_string = "$|0 \\rangle$-initialization"
    else:
        underline_states(ax, list_of_states_to_underline, fontsize=N + 4)
        init_string = ""

    ############################### TITLE #################################
    # Set the title for the plot.
    if title is None:
        # Construct subplot title from the parameters.
        dictstring = {
            "n": f"$n={n}$", "i": f"Instance #{instance}", "init": f"{init_string}",
            "p": f"$p={p}$", "ra": f"$ra={random_attempts}$", "k": f"$k={k}$"
        }
        title_string = ', '.join([dictstring[x] for x in dictstring if x not in dont_show_in_title])
        
        # Append bounds and parameters information to the title.
        bounds_and_pars0 = datafile.split('pars0')[1].split('_data')[0]
        bounds_and_pars0 = bounds_and_pars0.replace("pi", "\pi").replace("x", "\\times")
        title_string += f"\n${bounds_and_pars0}$"
    
        ax.set_title(title_string, fontsize=N)
    else:
        ax.set_title(title, fontsize=N)

    ##################################################################
    
    # Set y-axis limits and label.
    ax.set_ylim(0, 103)
    ax.set_ylabel("percentage [%]")
    
    # Display grid and minor ticks for better readability.
    plt.minorticks_on()
    plt.grid(alpha=0.2)

    # Set legend for the layers.
    legend = [f"layer {layer}" for layer in range(1, maxp + 1)]
    ax.legend(legend)
    
    # Show the plot.
    plt.show()


# In[ ]:


# def plot_list_of_files_parameter_fixing(path, datafiles, associated_files, init_name, title=None,
#                        dont_show_in_titles=[],  figsize=(16, 10), N=10, dpi=300):
#     """
#     """
#     # Numero di subplot richiesti
#     n_subplots = len(datafiles)

#     # Imposta il numero di righe e colonne per i subplot
#     n_rows = 2
#     n_cols = (n_subplots + 1) // n_rows

#     # Crea la figura
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
#     fig.suptitle(title, fontsize=N + 2)

#     # Flatten dell'array di assi per accesso sequenziale
#     axes = axes.flatten()

#     # Loop attraverso ogni datafile e i suoi file associati
#     for idx, (datafile, associated_files) in enumerate(zip(datafiles, associated_files)):

#         ax = axes[idx]

#         ################################ ESTRAI PARAMETRI ################################
#         n, instance, init_name, maxp, random_attempts, k = define_parameters_from_filename(path+datafile, verbose=False)
#         plot_title = f"n={n}, i={instance}, init={init_name}, maxp={maxp}, ra={random_attempts}, k={k}"

#         U, subsets_dict = define_instance(n, instance, verbose=False)
#         states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k)
        
#         ##################################################################################
#         df_final = None

#         # Per ogni file associato
#         for file_to_read in associated_files:
#             file_to_read = path + file_to_read
#             df = pd.read_csv(file_to_read, dtype={'states': "str"})

#             if df_final is None:
#                 df_final = df
#             else:
#                 df_final = pd.merge(df_final, df, on="states", how="outer")

#         # Riempimento e normalizzazione
#         df_final = df_final.fillna(0)
#         for p in range(1, maxp + 1):
#             column_name = f'counts_p{p}'
#             total = df_final[column_name].sum()
#             if total > 0:
#                 df_final[column_name] = (df_final[column_name] / total) * 100

#          # Ordina le colonne in base a "p" preservando l'ordine numerico
#         sorted_columns = sorted(
#             [col for col in df_final.columns if col.startswith('counts_p')],
#             key=lambda x: int(x.split('p')[1])
#         )

#         ################################ PLOT ################################
#         ax.set_title(plot_title, fontsize=N)

#         colors = ["darkorange", "crimson", "indigo"]
#         df_final.plot(
#             x='states', kind="bar", width=0.7, fontsize=N, stacked=False, ax=ax, legend=True,
#             color=colors, alpha=1
#         )

#         ################################ Etichette sulle barre ###############################
#         for state in df_final['states']:
#             max_height = 0
#             label_text = ""
#             x_position = None

#             for i, container in enumerate(ax.containers):
#                 p = i + 1
#                 row = df_final[df_final['states'] == state]
#                 if not row.empty:
#                     value = row[f'counts_p{p}'].values[0]
#                     if value > max_height:
#                         max_height = value
#                         label_text = f"{value:.1f}%"
#                         x_position = container[df_final[df_final['states'] == state].index[0]].get_x() + container[0].get_width() / 2

#             if max_height > 0:
#                 ax.text(
#                     x_position, max_height,
#                     label_text, ha='center', va='bottom', fontsize=8, weight='bold'
#                 )

#         ############################### EVIDENZIA E SOTTOLINEA ###############################
#         highlight_correct_ticks(ax, EXACT_COVERS)

#         # Sottolinea gli stati iniziali
#         if init_name == "all1":
#             one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
#             init_string = "$|1 \\rangle$-initialization"
#             underline_states(ax, one_one_states, fontsize=N+2)
#         elif init_name == "all0":
#             underline_states(ax, "0" * n, fontsize=N+2)
#             init_string = "$|0 \\rangle$-initialization"
#         else:
#             underline_states(ax, list_of_states_to_underline, fontsize=N+4)
#             init_string = ""


#         ############################### TITLE #################################
#         # Construct subplot title.
#         dictstring = {"n":f"$n={n}$", "i":f"Instance \#{instance}", "init":f"{init_string}", 
#                       "p":f"$p={p}$", "ra":f"$ra={random_attempts}$", "k":f"$k={k}$"}
#         title_string = ', '.join([dictstring[x] for x in dictstring if x not in dont_show_in_titles])
        
#         # Append bounds information to the title.
#         bounds_and_pars0 = datafile.split('pars0')[1].split('_data')[0]
#         bounds_and_pars0 = bounds_and_pars0.replace("pi", "\pi").replace("x", "\\times")
#         title_string += f"\n${bounds_and_pars0}$"

#         ax.set_title(title_string, fontsize=N)
#         ##################################################################
        
#         ax.set_ylim(0, 103)
#         plt.minorticks_on()
#         plt.grid(alpha=0.2)

#         if idx == 0 or idx == 5: 
#             ax.set_ylabel("percentage [%]")
#         else:
#             ax.set_ylabel("")

#         # Legenda
#         legend = [f"layer {layer}" for layer in range(1, maxp+1)]
#         ax.legend(legend)


#     # Nascondi gli assi inutilizzati
#     for idx in range(n_subplots, len(axes)):
#         fig.delaxes(axes[idx])
    
#     plt.subplots_adjust(wspace=0.17, hspace=0.5, left=0.03, right=0.97)
#     plt.show()

