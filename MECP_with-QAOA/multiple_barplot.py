"""
Module for creating multiple bar plots.

This module includes functions to generate bar plots with multiple data groups,
using the Matplotlib library. The code is designed to support clear and customizable
data visualizations.
"""
import math
from typing import List, Dict, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from utils_to_study_an_instance import define_instance, find_spectrum
from utils_for_plotting_and_reading import define_parameters_from_filename
from utils_for_plotting_and_reading import find_files_containing_string
from utils_for_plotting_and_reading import highlight_correct_ticks

def custom_sort(file_name: str, string_order: List[str]) -> int:
    """
    Returns the index of the first matching string from `string_order` in `file_name`.
    If no match is found, returns the length of `string_order`, placing unmatched files at the end.

    Parameters:
    file_name (str): The file name to be sorted.
    string_order (list[str]): The order of strings to match.

    Returns:
    int: The index of the first match or the length of `string_order` if no match.
    """
    for idx, key in enumerate(string_order):
        if key in file_name:
            return idx
    return len(string_order)  # Place unmatched files at the end

def from_file_to_percentage(filename: str,
                            data_filename: str,
                            p: int,
                            random_attempts: int) -> pd.DataFrame:
    """
    Computes the percentage of each value in the CSV file relative to the total,
    along with the average, max, and min values per row. It also identifies the best
    histogram index from a metadata file and sorts the data based on that index.

    Parameters:
    filename (str): Path to the CSV file containing the data.
    data_filename (str): Path to the metadata file used to identify the best histogram index.
    p (int): The parameter `p` used in the best histogram column name.
    random_attempts (int): The number of random attempts used in the best histogram column name.

    Returns:
    pd.DataFrame: A DataFrame with the percentage of values, average, max, and min,
                  sorted based on the best histogram index.
    """
    # Load data from CSV file
    df = pd.read_csv(filename, dtype=str).set_index('states')
    df = df.astype(float).fillna(0.0)

    # Compute percentages and additional columns
    total = df.sum()
    percentage = (df / total) * 100
    percentage['average'] = percentage.mean(numeric_only=True, axis=1)
    percentage['max'] = percentage[percentage.columns[:-1]].max(axis=1)
    percentage['min'] = percentage[percentage.columns[:-1]].min(axis=1)

    # Find the best histogram index from metadata file
    with open(data_filename, 'r') as data_file:
        for line in data_file:
            if 'Attempt that reached the best result with' in line:
                string = line.split('#')[1]
                i_best = string.split(' ')[0]

    column_best = f'counts_p{p}_{i_best}of{random_attempts}'
    percentage = percentage[[column_best, "average", "max", "min"]]
    percentage = percentage.sort_values(column_best, ascending=False)

    return percentage

def extract_parameters_and_ec(filename_list: List[str]
                              ) -> Tuple[Tuple[int, int, str, float, int, int], List[str], str]:
    """
    Extracts parameters and computes the Minimum Exact Cover (MEC) from a given file.
    This function defines the parameters from the filename, retrieves the necessary instance data,
    and computes the exact covers and the minimum exact cover (MEC) based on a specific instance.

    Parameters:
    - filename_list (list of str): 
        A list containing file names (expected to have at least one filename).

    Returns:
    - Tuple:
        - parameters (tuple): A tuple containing extracted parameters from the filename:
            - n (int): A parameter related to the system size.
            - instance (int): An identifier for the specific instance.
            - init_name (str): The initialization type (e.g., "all1", "all0").
            - p (float): A probability parameter.
            - random_attempts (int): The number of random attempts.
            - k (int): A parameter related to the setting of k.
        - exact_covers (list of str): A list of exact covers.
        - mec (str): The Minimum Exact Cover (MEC) from the list of exact covers, 
                     based on the minimum number of '1's.

    Example:
    >>> extract_parameters_and_ec(['data_file_1.txt'])
    ((10, 1, "all1", 0.5, 10, 3), ['000011', '000101', '010000'], '000011')
    """

    parameters = define_parameters_from_filename(filename_list[0], verbose=False)
    n = parameters[0]
    instance = parameters[1]
    
    # Define the instance and get relevant data
    U, subsets_dict = define_instance(n, instance, verbose=False)
    
    # Find exact covers
    exact_covers = find_spectrum(U, subsets_dict, n, k=1)[5]
    
    # Compute the Minimum Exact Cover (mec)
    mec = min(exact_covers, key=lambda s: s.count('1')) 
    
    return parameters, exact_covers, mec
        
def set_titles(parameters: Tuple[int, int, str, float, int, int]) -> Tuple[str, str]:
    """
    Constructs and returns the subplot and figure titles for a plot based on the provided parameters.
    
    Parameters
    ----------
    parameters (tuple): A tuple containing the following elements:
        - n (int): A parameter related to the system size.
        - instance (int): An identifier for the specific instance.
        - init_name (str): The initialization type (e.g., "all1", "all0").
        - p (float): A probability parameter.
        - random_attempts (int): The number of random attempts.
        - k (int): A parameter related to the setting of k.
    dont_show_in_title (list of str):
        A list of keys that should be excluded from the subplot title.
    dont_show_in_titles (list of str):
        A list of keys that should be excluded from both the subplot and figure titles.

    Returns
    -------
    title_string (str): The subplot title with the relevant parameters.
    figure_title_string (str): The figure title with the relevant parameters.

    Example
    -------
    >>> set_titles((10, 1, "all1", 0.5, 10, 3), ["k"], ["p"])
    ('n=10, Instance #1, $|1 \\rangle$-initialization, ra=10',
     'Performance of different k settings\nn=10, Instance #1, $|1 \\rangle$-initialization, ra=10, k=3')
    """
    n, instance, init_name, p, random_attempts, k = parameters
    
    # Highlight initialization states in the plot
    if init_name == "all1":
        init_string = "$|1 \\rangle$-initialization"
    elif init_name == "all0":
        init_string = "$|0 \\rangle$-initialization"
    else:
        init_string = ""

    # Construct a dictionary of formatted strings
    dictstring = {"n": f"$n={n}$",
                  "i": f"Instance #{instance}",
                  "init": f"{init_string}",
                  "p": f"$p={p}$",
                  "ra": f"$ra={random_attempts}$",
                  "k": f"$k={k}$"
                  }
    
    dont_show_in_title = ["i", "k"]  # Keys to omit from the figure title
    dont_show_in_titles = ["n", "p", "ra", "k", "init"]  # Keys to omit from subplot titles

    # Create figure title string
    figure_title_string = "Performance of different k settings\n"
    figure_title_string += ", ".join(
        [dictstring[x] for x in dictstring if x not in dont_show_in_title])

    # Create subplot title string
    title_string = ', '.join([dictstring[x] for x in dictstring if x not in dont_show_in_titles])

    return title_string, figure_title_string

def customize_ax(ax: plt.Axes, fontsize: int, instance: int, num_cols: int) -> None:
    """
    Customizes the appearance of the plot axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to customize.
    - fontsize (int): The font size to use for tick labels and axis labels.
    - instance (int): The instance number used to conditionally set the y-axis label.
    - num_cols (int): The number of columns used to determine when to add the y-axis label.

    Returns:
    - None: This function modifies the axes in place.
    """
    ax.tick_params(axis='x', which='major', labelsize=fontsize - 1, rotation=90)
    ax.tick_params(axis='y', which='major', labelsize=fontsize)

    ax.set_xlabel("States", fontsize=fontsize + 1)
    if instance % num_cols == 1:
        ax.set_ylabel("Percentage [%]", fontsize=fontsize + 1)

    ax.set_xlim(xmin=-1)
    ax.set_ylim(ymin=0, ymax=110)
    ax.minorticks_on()
    ax.grid(alpha=0.2)


def multiple_barplot(path: str,
                     string_order: List[str],
                     color_map: Dict[str, str],
                     order_labels: List[str],
                     filename_to_save: str
                     ) -> None:
    """
    Generate multiple bar plots based on input data files,
    visualizing the performance of different settings.

    Args:
        path (str): The path where input files are located.
        string_order (list[str]): List of strings defining the sorting order of files.
        color_map (dict[str, str]): A dictionary mapping keys to colors for bar plots.
        order_labels (list[str]): Labels for the order in which bars are plotted.
        filename_to_save (str): File name to save the figure; if empty, the plot is not saved.

    Returns:
        None
    """

    fontsize = 12  # Font size for titles and labels
    num_rows = 5  # Number of rows for subplots
    num_cols = math.ceil(10 / num_rows)  # Calculate the number of columns

    # Initialize the figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12), dpi=300, squeeze=False)
    axes = axes.flatten()

    for instance in range(1, 11):
        print(instance)
        ax = axes[instance - 1]  # Select the corresponding subplot axis

        # Find files corresponding to the current instance
        substrings = [f"_mail{instance}_"]
        filename_list, data_filename_list = find_files_containing_string(substrings, path)

        # Sort the file lists
        filename_list = sorted(filename_list, key=lambda x: custom_sort(x, string_order))
        data_filename_list = sorted(data_filename_list, key=lambda x: custom_sort(x, string_order))

        # Extract parameters and instance details from file names            
        parameters, exact_covers, mec = extract_parameters_and_ec(filename_list[0])
        _, instance, _, p, random_attempts, _ = parameters
        
        # Keys for color mapping
        keys = list(color_map.keys())

        # Combine data from all files
        combined_data = []
        for (filename, data_filename) in zip(filename_list, data_filename_list):
            percentage = from_file_to_percentage(filename, data_filename, p, random_attempts)
            combined_data.append(percentage)

        # Merge DataFrames with MultiIndex and fill missing values with 0
        df = pd.concat(combined_data, axis=1, join='outer', keys=keys)
        df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]
        df = df.fillna(0)

        # Identify columns for counts, average, max, and min
        count_columns = [col for col in df.columns if '_counts' in col]
        average_columns = [col for col in df.columns if col.endswith('_average')]
        max_columns = [col for col in df.columns if col.endswith('_max')]
        min_columns = [col for col in df.columns if col.endswith('_min')]

        bar_width = 0.3  # Width of bars
        x_positions = range(len(df))  # Positions for bars on the x-axis

        # Create bar plots
        for idx, (count_col, avg_col, 
                  max_col, min_col) in enumerate(zip(count_columns, average_columns,
                                                     max_columns, min_columns)):
            
            offset = (idx - 1) * bar_width  # Offset for grouped bars
            color = color_map[keys[idx]]  # Get color from color_map

            # Create bar plot for counts
            ax.bar([x + offset for x in x_positions], df[count_col], color=color,
                   alpha=1, label=f"{order_labels[idx]}: {df[count_col].sum():.1f}%", width=bar_width)

            # Add error bars for mec and exact covers
            for pos, state in enumerate(df.index):
                if state == mec or state in exact_covers:
                    value = df.loc[state, avg_col]
                    max_value = df.loc[state, max_col]
                    min_value = df.loc[state, min_col]

                    lower_error = value - min_value
                    upper_error = max_value - value

                    ax.errorbar([pos + offset], [value],
                                yerr=[[lower_error], [upper_error]],
                                fmt='.', alpha=1, ecolor="k", color=color,
                                markerfacecolor=color, markeredgecolor='k', capthick=1, capsize=5, elinewidth=1,
                                zorder=2, markersize=8, linestyle='None')

        title_string, figure_title_string = set_titles(parameters)
        ax.set_title(title_string, fontsize=fontsize)
        fig.suptitle(figure_title_string, fontsize=fontsize+3)
        
        # Highlight exact covers in x-ticks
        highlight_correct_ticks(ax, exact_covers)
        customize_ax(ax, fontsize, instance, num_cols)
    
    # Add legend for the entire figure
    legend_handles = [mpatches.Patch(color=color, label=label)
                      for label, color in color_map.items()]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.05), borderaxespad=0.,
               fontsize=fontsize, ncol=len(order_labels))

    # Remove unused axes
    for i in range(10, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout() # Adjust layout

    # Save the figure if a file name is provided
    if filename_to_save:
        plt.savefig(filename_to_save, bbox_inches="tight")

    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    # Random parameters, all0 vs all1.
    multiple_barplot(path="./FILES_PER-IMMAGINI-LATEX/tutte-le-istanze_all0_all1_20ra/",
                     string_order=['all1', 'all0'],
                     order_labels=['$all1$', '$all0$'],
                     color_map={"$|0\\rangle$ initialization": 'crimson',
                                "$|1\\rangle$ initialization": 'steelblue'},
                     filename_to_save="all0_all1_random.pdf")

    # Random parameters, varying k.
    multiple_barplot(path="./FILES_PER-IMMAGINI-LATEX/varie_k_options_tol1e4_all1/",
                     string_order=['L=n', 'maxLEC', 'Lmec'],
                     order_labels=['$L=n$', '$L=$max($L_{EC}$)', '$L=L_{mec}$'],
                     color_map={'$L=n$': 'darkorange',
                                '$L=$max($L_{EC}$)': 'crimson',
                                '$L=L_{mec}$': 'steelblue'},
                     filename_to_save="performance_k.pdf")
    