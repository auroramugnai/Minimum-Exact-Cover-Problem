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
                                title: str = '',
                                fontsize: int = 13,
                                figsize: Tuple[int, int] = (7, 3)) -> Axes:
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
    fontsize : int, optional
        The fontsize to set in the figure.
    figsize : tuple, optional
        The figure size.

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
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x="states", y=column_to_plot, data=percentage, 
                     width=0.7, color='steelblue', alpha=0.5)
    
    # Add percentage labels on top of the bars
    labels = percentage[column_to_plot].round(1).astype('str') + '%'
    for container in ax.containers:
        ax.bar_label(container, labels=labels, fontsize=fontsize-3)
    
    ### Highlight exact covers' ticks and underline initial states
    df_for_ticks = percentage.copy()
    df_for_ticks["states"] = df_for_ticks.index 

    underline_states(plt.gca(), states_to_underline, fontsize=fontsize)  # Underline the specific states
    highlight_correct_ticks(plt.gca(), EXACT_COVERS)  # Highlight exact covers

    
    ### Refine plot aesthetics
    plt.xlabel("States", fontsize=fontsize)
    plt.ylabel("", fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2, rotation="vertical")  # Rotate x-axis labels
    plt.yticks(fontsize=fontsize)
    plt.xlim(xmin=-1)  
    plt.ylim(ymin=0, ymax=106)
    plt.minorticks_on()
    plt.grid(alpha=0.2, which='both') 
    plt.title(title, fontsize=fontsize)  

    return ax


#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_histogram_of_best_column(df: pd.DataFrame, 
                                  best_column: str, 
                                  EXACT_COVERS: List[str], 
                                  states_to_underline: List[str], 
                                  title: str = '',
                                  fontsize: int = 13,
                                  figsize: Tuple[int, int] = (7, 3)) -> None:
    """
    Plots the histogram of the best column from a dataframe, highlighting exact covers and underlining specific states.
    Also overlays error bars for the average values using the max-min error.

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
    fontsize : int, optional
        The fontsize to set in the figure.
    figsize : tuple, optional
        The figure size.

    Returns
    -------
    None
    """
    # Generate the basic histogram for the best column
    ax = plot_histogram_of_df_column(df, best_column, EXACT_COVERS, states_to_underline, 
                                     title=title, fontsize=fontsize, figsize=figsize)
    
    # Set the dataframe index to 'states' and ensure numeric conversion
    df = df.set_index('states').astype(float).fillna(0.0)

    # Compute percentages and add average, max, and min columns
    total = df.sum()
    percentage = (df / total) * 100
    
    # Calculate the average, maximum, and minimum for each state
    percentage['average'] = percentage.mean(numeric_only=True, axis=1)
    percentage['max'] = percentage.max(numeric_only=True, axis=1)
    percentage['min'] = percentage.min(numeric_only=True, axis=1)
    
    # Calculate the max-min error (max - min)
    percentage['error'] = percentage['max'] - percentage['min']
        
    # Keep only the best column, average, and error columns
    percentage = percentage[[best_column, "average", "error"]]
    percentage = percentage.sort_values(best_column, ascending=False)
    
    # Overlay error bars for the average values using max-min error
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]  # Get the x positions of bars
    y_coords = percentage["average"]  # Average values for the y-axis
    ax.errorbar(x=x_coords, y=y_coords, yerr=percentage["error"], linestyle="",
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
        The filename containing parameter details, such as "./dim6_mail5_all0_random_p3_10ra_k0.085_".
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
    >>> define_parameters_from_filename("./dim6_mail3_all0_random_p3_10ra_k0.067_", verbose=True)
    n=6, i=3, init=all0, p=3, ra=10, k=0.067
    """
    # Remove path information if present and isolate the filename
    if '/' in filename:
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
    if '_p' in filename:
        p = int(filename.split('_p')[1].split('_')[0])
    elif '_maxp_' in filename:
        p = int(filename.split('_maxp_')[1].split('_')[0])


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



def find_files_containing_string(path: str, strings: Optional[list] = [], verbose: bool = False) -> tuple:
    """
    Finds files that contain specific substrings in their names and sorts them based on instance.

    This function searches for files in the specified directory (`path`) that contain all substrings 
    in the `strings` list (or no substrings if the list is empty). It returns two lists: one for normal 
    files and one for data files (those containing '_data' in their filenames). Both lists are sorted 
    based on the instance number extracted from the filenames.

    Parameters
    ----------
    path : str
        The directory path where the files should be searched.
    strings : list of str, optional
        A list of substrings to search for in the file names. If an empty list is provided, 
        the condition will always evaluate to `True`, and all files will be considered.
    verbose : bool, optional
        If True, prints the found files and their associated data files (default is False).

    Returns
    -------
    tuple
        A tuple containing two lists:
        - FILENAME_list : list of str
            List of files that match the substring criteria, excluding '_data' in the filename.
        - DATA_FILENAME_list : list of str
            List of files that match the substring criteria and contain '_data' in the filename.
    
    Example
    -------
    >>> find_files_containing_string(['dim6', 'mail3'], './data/', verbose=True)
    This would search for files in './data/' that contain both 'dim6' and 'mail3' in their filenames, 
    and print the results if `verbose` is True.
    """
    # Initialize empty lists to store normal files and data files
    FILENAME_list = []
    DATA_FILENAME_list = []

    # Loop over all files in the specified directory
    for obj in os.listdir(path):
        file_path = os.path.join(path, obj)
        
        # Check if the object is a file and contains all substrings in `strings` 
        # (including the case when `strings` is empty)
        if os.path.isfile(file_path) and (not strings or np.all([s in obj for s in strings])):
            if '_data' in obj:
                DATA_FILENAME_list.append(file_path)
            else:
                FILENAME_list.append(file_path)

    if len(FILENAME_list) > 1:
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

def build_title(filename: str, 
                dont_show_in_title: Optional[list] = [],
                show_bounds: Optional[bool] = True) -> str:
    """
    Builds a formatted title string based on the parameters extracted from the given filename.

    The function extracts parameters such as `n`, `instance`, `init_name`, `p`, `random_attempts`, and `k` 
    from the provided `filename` using `define_parameters_from_filename`. It then constructs a title string 
    with those parameters and appends bounds information extracted from the filename. It also allows excluding 
    certain parameters from the title using the `dont_show_in_title` list.

    Parameters
    ----------
    filename : str
        The filename from which parameters will be extracted.
    dont_show_in_title : list of str, optional
        A list of parameters to exclude from the title string. Default is [].
    show_bounds : bool, optional
        If True, bounds are shown in the title. Defaults to True.

    Returns
    -------
    str
        A formatted string that can be used as a plot or graph title.

    Example
    -------
    >>> file = "dim6_mail1_all1_p3_2ra_k0.167_BOUNDS[0,2pi]x[-110,110]_pars0[0,2pi]x[-110,110].csv"
    >>> build_title(file, ["n", "k"])
    Instance #1, $p=3$, $ra=2$, [0,2pi]x[-110,110]$'
    """

    parameters = define_parameters_from_filename(filename, verbose=False)
    n, instance, init_name, p, random_attempts, k = parameters

    # Prepare the title string dictionary with the parameters
    d = {"n": f"$n={n}$", "i": f"Instance #{instance}", "init": f"{init_name}",
         "p": f"$p={p}$", "ra": f"$ra={random_attempts}$", "k": f"$k={k}$"}

    # Build the title string excluding the parameters in the `dont_show_in_title` list
    title_string = ', '.join([d[x] for x in d if x not in dont_show_in_title])

    if show_bounds:
        # Append bounds information to the title
        bounds_and_pars0 = filename.split('pars0')[1].split('_data')[0]
        bounds_and_pars0 = bounds_and_pars0.replace("pi", "\\pi").replace("x", "\\times")
        title_string += f"\n${bounds_and_pars0}$"

    return title_string


#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_file(FILENAME: str, DATA_FILENAME: str, colorchosen: str, alpha: float,
              states_to_underline: Optional[List[str]] = None, 
              show_title: Optional[bool] = True,
              dont_show_in_title: List[str] = [], 
              figsize: Tuple[int, int] = (18, 8), 
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
    show_title : bool, optional
        If True, a title is shown with file parameters. Default is True.
    dont_show_in_title : list of str, optional
        List of parameters not to show in the plot's title (default is empty list).
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

    # Extract parameters from filename 
    n, instance, init_name, p, random_attempts, k = define_parameters_from_filename(DATA_FILENAME, 
                                                                                    verbose=False)

    # Define the problem instance using the extracted parameters
    U, subsets_dict = define_instance(n, instance, verbose=False)

    # Analyze spectrum to extract relevant state information
    states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k=1)
                                             
    MEC = min(EXACT_COVERS, key=lambda s: s.count('1'))  # Minimum exact cover

    # Load the state data into a pandas DataFrame
    df = pd.read_csv(FILENAME, dtype=str).set_index('states')
    df = df.astype(float).fillna(0.0)

    # Compute percentages and add columns for average and range (max - min)
    total = df.sum()
    percentage = (df / total) * 100
    percentage['average'] = percentage.mean(numeric_only=True, axis=1)

    max_values = percentage[percentage.columns[:-1]].max(numeric_only=True, axis=1)
    min_values = percentage[percentage.columns[:-1]].min(numeric_only=True, axis=1)
    percentage['range'] = max_values - min_values

    # Identify the best histogram index based on metadata file
    with open(DATA_FILENAME, 'r') as DATA_FILE:
        for line in DATA_FILE:
            if 'Attempt that reached the best result with' in line:
                string = line.split('#')[1]
                i_best = string.split(' ')[0]

    # Select the column corresponding to the best result
    column_best = f'counts_p{p}_{i_best}of{random_attempts}'

    # Keep the best and average results for plotting
    percentage = percentage[[column_best, "average", "range"]]
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
    ax.errorbar(x=x_coords, y=y_coords, yerr=percentage["range"], linestyle="",
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

    
    # Set title
    title = build_title(FILENAME, dont_show_in_title)
    ax.set_title(title, fontsize=N)
    
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
                       show_title: Optional[bool] = True,
                       figsize: tuple = (8, 8), dpi: int = 100, N: int = 10):
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
    show_title: bool, optional
        If True, a figure-title is shown with file parameters.  Default is True.
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
        # Extract parameters from filename
        n, instance, init_name, p, random_attempts, k = define_parameters_from_filename(DATA_FILENAME, verbose=False)
            
        # Define the problem instance based on extracted parameters.
        U, subsets_dict = define_instance(n, instance, verbose=False)

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
        
        print(DATA_FILENAME)
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
                    markerfacecolor='none', linewidth=1, marker='o', 
                    markersize=3, color="k", ecolor="k", 
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

        # Set title
        subplot_title = build_title(FILENAME, dont_show_in_titles)
        ax.set_title(subplot_title, fontsize=N)
        
        # Set x and y labels
        if num_subplot%2 == 0:  
            ax.set_ylabel("percentage [%]", fontsize=N)
        else:
            ax.set_ylabel("")
        plt.xlabel("states", fontsize=N)

        # Refine plot aesthetics: set labels and grid.       
        plt.xticks(fontsize=N-1, rotation="vertical")
        plt.yticks(fontsize=N)
        plt.xlim(xmin=-1)
        plt.ylim(ymin=0, ymax=110)
        plt.minorticks_on()
        plt.grid(alpha=0.2)

    # Save the figure with a timestamp.
    current_datetime = datetime.now().strftime("@%Y-%m-%d@%Hh%Mm%Ss")
    plt.savefig(f"all1_random_{current_datetime}.pdf")
    
    # Set the overall title.
    if show_title:
        figure_title = build_title(FILENAME, dont_show_in_title, show_bounds=False)
        fig.suptitle(figure_title, fontsize=N+3)
        
    # Display the plot.
    plt.tight_layout()
    plt.show()



#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_file_parameter_fixing(datafile: str,
                               file: str,
                               show_title: Optional[bool] = True,
                               figsize: Tuple[int, int] = (16, 10),
                               N: int = 10,
                               dpi: int = 300
                               ) -> None:
    """
    Plots a bar chart based on the parameter counts from multiple associated files.
    
    This function reads data from the given `datafile` and associated files. 
    It then normalizes the counts and creates a stacked bar plot with 
    appropriate labels and formatting.

    Parameters:
        datafile (str): The main data file that contains parameter information.
        file (str): The associated CSV file containing data to plot.
        show_title (Optional[bool], optional): If True, a title with parameters is shown. Default is True.
        figsize (Tuple[int, int], optional): Figure size for the plot (default is (16, 10)).
        N (int, optional): Font size for the plot labels (default is 10).
        dpi (int, optional): Dots per inch for the plot resolution (default is 300).
    
    Returns:
        None: The function does not return any value but displays the plot.
    """
    
    # Create the figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    ################################ EXTRACT PARAMETERS ################################
    n, instance, init_name, maxp, random_attempts, k = define_parameters_from_filename(datafile, verbose=False)
    plot_title = f"n={n}, i={instance}, init={init_name}, maxp={maxp}, ra={random_attempts}, k={k}"

    U, subsets_dict = define_instance(n, instance, verbose=False)
    states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k)
    
    # Load data from CSV file
    df = pd.read_csv(file, dtype={'states': "str"}).set_index('states')
    df = df.astype(float).fillna(0.0)
    
    # Compute percentages and additional columns
    df = (df / df.sum()) * 100
    df = df.reset_index()  # Make 'states' a column again
    df = df.sort_values(f'counts_p{maxp}', ascending=False)  # Sort based on maxp column

    ################################ PLOT ################################
    colors = ["darkorange", "crimson", "indigo"]
    df.plot(
        x='states', kind="bar", width=0.7, fontsize=N, stacked=False, ax=ax, legend=True,
        color=colors, alpha=1
    )

    ################################ LABELS ON BARS ################################
    for i, state in enumerate(df['states']):
        # Find the value for the column f'counts_p{maxp}' for the current state
        row = df[df['states'] == state]
        if not row.empty:
            max_height = row[f'counts_p{maxp}'].values[0]
            if max_height > 0:
                # Get the position of the bar in the corresponding container
                container = ax.containers[maxp - 1]  # maxp corresponds to the container index (1-based)
                bar = container[i]  # Find the bar corresponding to position i
                x_position = bar.get_x() + bar.get_width() / 2

                # Add the label above the bar
                ax.text(
                    x_position, max_height,
                    f"{max_height:.1f}%", ha='center', va='bottom', fontsize=N
                )

    ############################### HIGHLIGHT AND UNDERLINE ###############################
    highlight_correct_ticks(ax, EXACT_COVERS)

    # Underline initial states
    if init_name == "all1":
        one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
        init_string = "$|1 \\rangle$-initialization"
        underline_states(ax, one_one_states, fontsize=N+2)
    elif init_name == "all0":
        underline_states(ax, "0" * n, fontsize=N+2)
        init_string = "$|0 \\rangle$-initialization"
    else:
        underline_states(ax, list_of_states_to_underline, fontsize=N+4)
        init_string = ""
        
    ##################################################################

    # Set title
    if show_title:
        title = build_title(datafile)    
        ax.set_title(title, fontsize=N)

    ax.set_ylim(0, 103)
    ax.set_ylabel("percentage [%]")
    plt.minorticks_on()
    plt.grid(alpha=0.2)

    # Legend
    legend = [f"Counts p={layer}" for layer in range(1, maxp + 1)]
    ax.legend(legend)
    plt.show()
    
#############################################################################################################
#############################################################################################################
#############################################################################################################
def plot_list_of_files_parameter_fixing(
    datafiles: List[str],
    associated_files: List[List[str]],
    dont_show_in_title: List[str] = [],
    dont_show_in_titles: List[str] = [],
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 300,
    N: int = 10,
    show_title: bool = True
) -> None:
    """
    Plots a series of data files with associated files, and saves and displays the plot.

    This function generates a series of subplots, each one corresponding to a data file and its
    associated files. It normalizes the counts, creates a bar plot, and displays it with labels and
    highlights based on the initialization scheme. The function also saves the resulting plot as a PDF.

    Parameters:
        datafiles (List[str]): A list of data file names to be plotted.
        associated_files (List[List[str]]): A list of lists, where each list contains associated files for the corresponding datafile.
        dont_show_in_title (List[str], optional): Parameters to exclude from the title of the subplot. Defaults to [].
        dont_show_in_titles (List[str], optional): Parameters to exclude from the overall figure title. Defaults to [].
        figsize (Tuple[int, int], optional): Size of the figure (default is (16, 10)).
        dpi (int, optional): Dots per inch for the plot resolution (default is 300).
        N (int, optional): Font size for the plot labels (default is 10).
        show_title (bool, optional): Whether to display the overall title of the figure. Defaults to True.

    Returns:
        None: The function does not return any value but displays the plot.
    """
    # Numero di subplot richiesti
    n_subplots = len(datafiles)

    # Imposta il numero di righe e colonne per i subplot
    n_cols = 2
    n_rows = math.ceil(n_subplots / n_cols)
    
    # Crea la figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)

    # Flatten dell'array di assi per accesso sequenziale
    axes = axes.flatten()

    # Loop attraverso ogni datafile e i suoi file associati
    for num_subplot, (datafile, file) in enumerate(zip(datafiles, associated_files)):

        ax = axes[num_subplot]

        ################################ ESTRAI PARAMETRI ################################
        n, instance, init_name, maxp, random_attempts, k = define_parameters_from_filename(datafile, verbose=False)
        plot_title = f"n={n}, i={instance}, init={init_name}, maxp={maxp}, ra={random_attempts}, k={k}"

        U, subsets_dict = define_instance(n, instance, verbose=False)
        states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k)
        
        ##################################################################################
        # Load data from CSV file
        df = pd.read_csv(file, dtype={'states': "str"}).set_index('states')
        df = df.astype(float).fillna(0.0)

        # Compute percentages and additional columns
        df = (df / df.sum()) * 100
        df = df.reset_index()  # Make 'states' a column again
        df = df.sort_values(f'counts_p{maxp}', ascending=False)  # Sort based on maxp column

        ################################ PLOT ################################
        # ax.set_title(plot_title, fontsize=N)

        colors = ["darkorange", "crimson", "indigo"]
        df.plot(
            x='states', kind="bar", width=0.7, fontsize=N, stacked=False, ax=ax,
            color=colors, alpha=1, legend=False
        )

        ################################ LABELS ON BARS ################################
        for i, state in enumerate(df['states']):
            # Find the value for the column f'counts_p{maxp}' for the current state
            row = df[df['states'] == state]
            if not row.empty:
                max_height = row[f'counts_p{maxp}'].values[0]
                if max_height > 0:
                    # Get the position of the bar in the corresponding container
                    container = ax.containers[maxp - 1]  # maxp corresponds to the container index (1-based)
                    bar = container[i]  # Find the bar corresponding to position i
                    x_position = bar.get_x() + bar.get_width() / 2

                    # Add the label above the bar
                    ax.text(
                        x_position, max_height,
                        f"{max_height:.1f}%", ha='center', va='bottom', fontsize=N
                    )
        ############################### EVIDENZIA E SOTTOLINEA ###############################
        highlight_correct_ticks(ax, EXACT_COVERS)

        # Sottolinea gli stati iniziali
        if init_name == "all1":
            one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
            init_string = "$|1 \\rangle$-initialization"
            underline_states(ax, one_one_states, fontsize=N+1)
        elif init_name == "all0":
            underline_states(ax, "0" * n, fontsize=N+1)
            init_string = "$|0 \\rangle$-initialization"
        else:
            underline_states(ax, list_of_states_to_underline, fontsize=N+1)
            init_string = ""


        # Set title
        subplot_title = build_title(file, dont_show_in_titles)
        ax.set_title(subplot_title, fontsize=N)  
        
        # Set x and y label
        ax.set_xlabel("states", fontsize=N)
        if num_subplot%2 == 0: 
            ax.set_ylabel("percentage [%]", fontsize=N)
        else:
            ax.set_ylabel("")
            
        # Refine plot aesthetics.
        ax.tick_params(axis='x', which='major', labelsize=N-3, rotation=90)
        ax.tick_params(axis='y', which='major', labelsize=N)
        # ax.set_xlim(xmin=-1, xmax= len(states_feasible))
        ax.set_ylim(ymin=0, ymax=120)
        ax.tick_params(axis='both', which='minor', bottom=False)
        ax.grid(alpha=0.2)

    # Nascondi gli assi inutilizzati
    for num_subplot in range(n_subplots, len(axes)):
        fig.delaxes(axes[num_subplot])

    # Aggiungi la legenda sopra i subplot (in cima alla figura)
    legend_labels = [f"Counts p={layer}" for layer in range(1, maxp + 1)]
    fig.legend(legend_labels, loc='upper center', 
               ncol=maxp, fontsize=N, bbox_to_anchor=(0.5, 1.03))  # Position legend above the figure
    
    # Set the overall title.
    if show_title:
        figure_title = build_title(file, dont_show_in_title, show_bounds=False)
        fig.suptitle(figure_title, fontsize=N+3)
        
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Ensure space for the legend

#     Save the figure with tight bounding box and padding
#     current_datetime = datetime.now().strftime("@%Y-%m-%d@%Hh%Mm%Ss")
#     plt.savefig(f"parameters_fixing_{init_name}.pdf", bbox_inches='tight', pad_inches=0.1)
    
    plt.show()