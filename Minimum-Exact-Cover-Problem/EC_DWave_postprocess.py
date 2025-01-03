import csv
import ast
import re
import os

import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from termcolor import colored # to color a dataframe

from instances import all_solutions


def plot_accuracy(accuracy_EC_values, accuracy_MEC_values, instances, NUNITS, NREADS, NSAMPLES):
    """
    Plots the accuracy of two metrics (num_EC / NREADS x NUNITS and num_MEC / NREADS x NUNITS) 
    for different instances, displaying the results in a line plot with labeled data points.
    
    Parameters
    ----------
    accuracy_EC_values (list of float): List of accuracy values for num_EC / NREADS x NUNITS.
    accuracy_MEC_values (list of float): List of accuracy values for num_MEC / NREADS x NUNITS.
    instances (list of int): List of instance numbers corresponding to the data.
    NUNITS (int): The number of units used in the experiment.
    NREADS (int): The number of reads in the experiment.
    NSAMPLES (int): The number of samples used in the experiment.
    
    Returns
    -------
    None: This function directly displays a plot and does not return any value.
    
    Notes
    -----
    - The plot includes two lines: one for the EC accuracy values and one for the MEC accuracy values.
    - The plot labels each data point with its corresponding accuracy value.
    """
    
    plt.rcParams.update({'font.family': 'Sans-serif', 'font.size': 14})

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.title(f"Accuracy\n NUNITS = {NUNITS}, NREADS = {NREADS}, NSAMPLES = {NSAMPLES}")

    plt.plot(instances, accuracy_EC_values, label='num_EC / NREADS x NUNITS', 
             marker='x', color='r')
    plt.plot(instances, accuracy_MEC_values, label='num_MEC / NREADS x NUNITS', 
             marker='x', color='b')

    # Adding text above each point
    for (i, value_EC, value_MEC) in zip(instances, accuracy_EC_values, accuracy_MEC_values):
        plt.text(
            i, value_EC, f'{value_EC:.2f}', 
            ha='center', 
            transform=plt.gca().transData,  # Use data coordinates for the position
            va='bottom',  # Align text at the bottom
            clip_on=True,  # Prevent text from overlapping axes
            bbox=dict(pad=1, alpha=0)  # Ensure the text isn't clipped
        )
        plt.text(
            i, value_MEC, f'{value_MEC:.2f}', 
            ha='center',
            transform=plt.gca().transData,  # Use data coordinates for the position
            va='bottom',  # Align text at the bottom
            clip_on=True,  # Prevent text from overlapping axes
            bbox=dict(pad=1, alpha=0)  # Ensure the text isn't clipped
        )

    # Set x ticks
    plt.xticks(instances)  # Set x-ticks to these natural numbers

    plt.xlabel('Instance')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.grid(True)

    plt.show()


def read_custom_csv(file_path):
    """
    Reads a CSV file with the specific format:
    array_to_num,counts,is_feasible
    Example row: [0, 2, 3],2,True

    Parameters
    ----------
    file_path (str): Path to the CSV file.

    Returns
    -------
    list of dict: A list of dictionaries with parsed data from the CSV file.
        Each dictionary contains:
        - 'state' (list): The list of numbers from the 'state' column.
        - 'label' (str): The label associated with the entry.
        - 'num_occurrences' (int): The number of occurrences from the 'num_occurrences' column.
    """
    data = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            parsed_row = {
                'state': ast.literal_eval(row['state']),
                'label': row['label'],
                'num_occurrences': int(row['num_occurrences'])
            }
            data.append(parsed_row)

    return data


def extract_filename_data(filename):
    """
    Extracts specific data from a filename with the format:
    MM-DD@HHhMMmSSs_EC_instanceX_NUNITSX_NREADSX_NSAMPLESX.csv

    Parameters
    ----------
    filename (str): The filename string to extract data from.

    Returns
    -------
    dict: A dictionary with the extracted data, including:
        - 'date' (str): The date in MM-DD format.
        - 'time' (str): The time in HHhMMmSSs format.
        - 'instance' (int): The instance number extracted from the filename.
        - 'NUNITS' (int): The number of units extracted from the filename.
        - 'NREADS' (int): The number of reads extracted from the filename.
        - 'NSAMPLES' (int): The number of samples extracted from the filename.

    Raises
    ------
    ValueError: If the filename does not match the expected format.
    """
    pattern = r"(?P<date>\d{2}-\d{2})@(?P<time>\d{2}h\d{2}m\d{2}s)_EC_instance(?P<instance>\d+)_NUNITS(?P<NUNITS>\d+)_NREADS(?P<NREADS>\d+)_NSAMPLES(?P<NSAMPLES>\d+)\.csv"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError("Filename does not match the expected format")

    extracted_data = {
        'date': match.group('date'),
        'time': match.group('time'),
        'instance': int(match.group('instance')),
        'NUNITS': int(match.group('NUNITS')),
        'NREADS': int(match.group('NREADS')),
        'NSAMPLES': int(match.group('NSAMPLES'))
    }
    return extracted_data


def process_files_in_directory(directory_path):
    """
    Processes all CSV files in the specified directory and extracts relevant information 
    from the file names and the CSV content.
    
    Parameters
    ----------
    directory_path (str): The path to the directory containing the files to be processed.
        
    Returns
    -------
    results (list of dict): A list of dictionaries, where each dictionary contains:
        - 'instance' (int): The extracted instance identifier from the file name.
        - 'filename_data' (dict): The data extracted from the file name (via `extract_filename_data`).
        - 'csv_data' (list): The parsed content of the CSV file (via `read_custom_csv`).
        
    Notes
    -----
    - Only files with a `.csv` extension are processed.
    - If a file name does not conform to the expected format, it will be skipped with a warning.
    - Each processed CSV file is associated with an "instance" key, which is extracted from the file name.
    """
    results = []
    instance_files = {}

    print("\nFiles in the chosen directory:")

    # Iterates through all files in the directory
    for file_name in os.listdir(directory_path):
        print(file_name)  
        if file_name.endswith(".csv"):  # Filters for only CSV files
            try:
                # Extracts data from the file name
                extracted_data = extract_filename_data(file_name)
                instance = extracted_data['instance']
                instance_files[instance] = {
                    'file_name': file_name,
                    'extracted_data': extracted_data
                }
            except ValueError:
                # Skips the file if it does not match the expected format
                print(f"Skipping file due to incorrect format: {file_name}")
                continue

    # Processes the files and adds the data to the result list
    for instance, file_info in instance_files.items():
        file_path = os.path.join(directory_path, file_info['file_name'])
        csv_data = read_custom_csv(file_path)
        results.append({
            'instance': instance,
            'filename_data': file_info['extracted_data'],
            'csv_data': csv_data
        })

    # Sort by increasing instance
    results.sort(key=lambda x: int(x['instance']))

    return results


def print_dictionary(d):
    """
    Prints the contents of a dictionary.

    Parameters
    ----------
    d (dict): The dictionary to print.

    Returns
    -------
    None: This function prints the key-value pairs in the dictionary.
    """
    for key, value in d.items():
        print(f"{key}: {value}")


def get_labels_from_directory(directory_path):
    """
    Returns a list of folder names (labels) in the specified directory.
    
    Parameters
    ----------
    directory_path (str): The path to the directory containing the subfolders.
    
    Returns
    -------
    list of str: A list of folder names present in the directory.
    """
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]



# ************************************************************************
# *************************** MAIN ***************************************
# ************************************************************************
if __name__ == "__main__":
    
    # Otteniamo tutte le cartelle presenti nella directory
    labels = get_labels_from_directory("./")
    labels.remove("__pycache__")
    
    # Se ci sono delle etichette, permettiamo all'utente di sceglierne una
    if len(labels) == 1:
        selected_label = labels[0]
        all_data = process_files_in_directory(selected_label)

    elif labels:
        print("Seleziona una delle seguenti etichette (cartelle):")
        for i, label in enumerate(labels, 1):
            print(f"{i}. {label}")
        
        # Chiediamo all'utente di scegliere un'etichetta
        while True:
            try:
                user_choice = int(input(f"Scegli un numero da 1 a {len(labels)}: "))
                if 1 <= user_choice <= len(labels):
                    selected_label = labels[user_choice - 1]
                    print(f"\nHai selezionato la cartella: {selected_label}")
                    break
                else:
                    print("Scelta non valida, per favore riprova.")
            except ValueError:
                print("Per favore inserisci un numero valido.")
        
        # Elaboriamo i file con la label scelta
        all_data = process_files_in_directory(selected_label)

    else:
        print("Nessuna cartella trovata nella directory.")


    # ----------------------------- PRINT DATA -----------------------------------
    instances = []
    for data in all_data:
        print("-" * 42)

        print("Instance:", data['instance'])
        instances.append(data['instance'])
        print_dictionary(data['filename_data'])

        #----------    Print the dataframe with colors   ------------
        solutions_str = [str(s) for s in all_solutions[data['instance']]]
        df = pd.DataFrame(data['csv_data'], dtype=str)
    
        # Define a function that colors arrays in green if they are a solution.
        color_solutions = lambda x: (colored(x, None, 'on_green') 
                                     if x in solutions_str
                                     else colored(x, 'white', None))

        df['state'] = df['state'].map(color_solutions)

        # Reset columns or they will be shifted.
        df.columns =  [colored('state', 'white', None)] + ["label", "num_occurrences"]

        print(df)    

    # ----------------------------- ACCURACY PLOT -----------------------------------

    # Costruisci il percorso completo del file JSON
    json_file_path = os.path.join(selected_label, f'{selected_label}_accuracy_values.json')
    
    # Verifica se il file esiste
    if not os.path.exists(json_file_path):
        print(f"Il file {json_file_path} non è stato trovato.")
    else:
        # Caricamento dal file JSON dei dati sull'accuracy
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            # Moltiplicazione di ciascun valore per 100
            accuracy_EC_values = [value * 100 for value in data['accuracy_EC_values']]
            print(accuracy_EC_values)
            accuracy_MEC_values = [value * 100 for value in data['accuracy_MEC_values']]
            print(accuracy_MEC_values)

        # Plot dei dati sull'accuracy
        NUNITS = all_data[0]["filename_data"]["NUNITS"]
        NREADS = all_data[0]["filename_data"]["NREADS"]
        NSAMPLES = all_data[0]["filename_data"]["NSAMPLES"]

        print(instances)
        plot_accuracy(accuracy_EC_values, accuracy_MEC_values, instances, 
                      NUNITS, NREADS, NSAMPLES)

