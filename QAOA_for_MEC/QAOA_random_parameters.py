#!/usr/bin/env python
# coding: utf-8

# # Random Parameters Minimization

# ### Import.

# In[1]:


from __future__ import annotations
from utils_to_build_QAOAAnsatz import *


# #### Funzioni per trovare k.

# In[4]:


def truncate(number, n):
    """
    if number has more than n digits -> returns only n digits
    if number is an int              -> returns number
    if number has less than n digits -> returns number
    """
    str_number = str(number)
    
    if not "." in str_number: return number
        
    decimals = str_number.split(".")[1]    
    if len(decimals) == n or len(decimals) < n:
        return number
    else:
        a = int(decimals[n])
        new_number = number - a / (10)**(n+1)
        truncated_number = round(new_number, n)
        return truncated_number
        
def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier  


# #### Dizionario contenente tutte le k riferite a dimensione 6

# In[5]:


k_dict = {'L=n': [0.3333333333333333,   0.5,   0.5,   0.3333333333333333,   0.5,   0.5,   0.5,   
                  0.3333333333333333,   0.25,   0.25],
          'L=max(L_EC)': [0.16666666666666666,   0.16666666666666666,   0.25,   0.16666666666666666,  
                          0.3333333333333333, 0.25,   0.25,   0.16666666666666666,   0.08333333333333333,  
                          0.08333333333333333],
          'L=L_MEC': [0.16666666666666666,   0.16666666666666666,   0.16666666666666666,
                      0.1111111111111111,   0.16666666666666666,   0.16666666666666666,   0.25,   
                      0.16666666666666666,  0.08333333333333333,   0.08333333333333333]}

# Arrotondo per eccesso
k_dict_new = {}
for key,value_list in k_dict.items():
    k_dict_new[key] = [round_up(v, 3) for v in value_list]
k_dict_new


# ### Minimization.

# In[ ]:


# p = 3 # number of layers
# random_attempts = 20
# init_string = "all1"
p = 1 # number of layers
random_attempts = 1
init_string = "all1"


# In[ ]:


from datetime import datetime
# import itertools
# import math 
# import os
# import random
import time
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator, StatevectorSampler


# In[ ]:


for instance in [1,2]:
    h = k_dict_new['L=max(L_EC)'][instance-1]
    print("*"*50)
    print(f"Instance {instance} with h = {h}\n")

    
    ### These will contain the files' names.
    FILENAME_list = [] # list of .csv containing the final histograms data.
    DATA_FILENAME_list = [] # list of .txt containing extra data such as energies, betas, gammas.

    
    ### Define the instance.
    U, subsets_dict = define_instance(n, instance, verbose=False)
    subsets = list(subsets_dict.values())
    _, _, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, h)
    MEC = [state for state in EXACT_COVERS if state.count("1") == min([x.count("1")  for x in EXACT_COVERS])]

    # show_spectrum(n, instance, h)

    
    ###########################################################
    
    ### Choose the initialization.
    if init_string == 'all1':
        # Only "1"-states.
        one_one_states = ["".join(elem) for elem in distinct_permutations('0'*(n-1) + '1')]
        init_name = one_one_states

    elif init_string == 'all0':
        init_name = ["000000"]
        print("init_name:", init_name)
    
    ###########################################################    
      
    ### Prepare the cost and mixing circuit.
    constant, hamiltonian, qc_cost = build_cost_circuit(n, instance, h, verbose=False)
    qc_mixing = build_mixing_circuit(n, instance, verbose=False)
    qc_initial = build_initialization_circuit(n, instance, init_name, verbose=False)
    print(f"Initialization: {init_string}")
    
    ###########################################################
    ###########################################################
    ### SET INITIAL PARAMETERS FOR MINIMIZATION AND BOUNDS
    
    gamma_bound = find_gamma_bound(n, instance, h, verbose=False)
    
    beta_0 = (0, 2*np.pi)
    gamma_0 =  (-gamma_bound, gamma_bound)
    string_0 = f"[0,2pi]x[-{gamma_bound},{gamma_bound}]"
    
    bnds_beta = (0, 2*np.pi)
    bnds_gamma = (-gamma_bound, gamma_bound)
    bnds_string = f"[0,2pi]x[-{gamma_bound},{gamma_bound}]"
    
    ###########################################################
    ###########################################################
    ### BUILD FILES' NAMES
    
    # current_datetime = datetime.now().strftime("@%Y-%m-%d@%Hh%Mm%Ss")
    current_datetime = datetime.now().strftime("%d-%m")
    
    header = f"PROVA@{current_datetime}_dim{n}_mail{instance}_{init_string}_random_p{p}_{random_attempts}ra_k{h}"
    header = header + f"_BOUNDS{bnds_string}_pars0{string_0}"
    # header = header + f"_BOUNDS_pars0_monotonic"
    print(header)
    
    DATA_FILENAME = header + '_data.txt'
    FILENAME = header + ".csv"
    print("FILENAME:", FILENAME)

    FILENAME_list.append(FILENAME)
    DATA_FILENAME_list.append(DATA_FILENAME)
    
    ###########################################################
    ###########################################################
    ### DO THE MINIMIZATION
    
    
    with open(DATA_FILENAME, 'a') as DATA_FILE:
        DATA_FILE.write(f"current datetime = {datetime.now().strftime('@%Y-%m-%d@%Hh%Mm%Ss')}")
        DATA_FILE.write(f"\np={p}\n")
        DATA_FILE.write(f"\ninit_string: {init_string}\n")
        DATA_FILE.write(f"\ninit_name: {init_name}\n")
    
        E_best = 100
        counter = 0
        TOTAL_start_time = time.time()
    
    
        for i in range(1,random_attempts+1):
            print(f"\n----- {i}/{random_attempts} random_attempts -----\n")
            DATA_FILE.write(f"\n----- {i}/{random_attempts} random_attempts -----\n")

            ### Build QAOAAnsatz.
            cost_vs_iteration = []
            ansatz = QAOAAnsatz(qc_cost, mixer_operator=qc_mixing, initial_state=qc_initial, reps=p, name='my_QAOA_circuit')

            ### Generate a pass manager without providing a backend.
            pm = generate_preset_pass_manager(optimization_level=3)
            ansatz_isa = pm.run(ansatz)
            hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)
        
            estimator = StatevectorEstimator()
            sampler = StatevectorSampler()

            
            # *******************************  MINIMIZE  *********************************

            ########################
            ### Do the minimization.
            pars_0 = [random.uniform(*beta_0) for _ in range(p)] + [random.uniform(*gamma_0) for _ in range(p)]  
            bnds = [bnds_beta]*p + [bnds_gamma]*p

            # pars_0_b, pars_0_g, list_of_beta_bnds, list_of_gamma_bnds = find_monotonic_pars_0(beta_0, gamma_0)
            # pars_0 = pars_0_b + pars_0_g
            # bnds = list_of_beta_bnds + list_of_gamma_bnds
            
            print(f"pars_0 = {pars_0}")
            print(f"bnds = {bnds}")
            ########################
            
            ### If you wnat to plot iterations you should choose cost_func_plot
            
            # res = minimize(cost_func, pars_0,
            #                args=(ansatz_isa, hamiltonian_isa, estimator), 
            #                method="COBYLA", options={"maxiter":200, "tol":1e-3})
            res = minimize(cost_func, pars_0, bounds=bnds,
                           args=(ansatz_isa, hamiltonian_isa, estimator), 
                           method="Nelder-Mead", options={"disp": True, "maxiter": 1200, "maxfev": 1200}, tol=1e-4)
            # res = minimize(cost_func, pars_0, bounds=bnds,
            #                args=(ansatz_isa, hamiltonian_isa, estimator), 
            #                method="Nelder-Mead", tol=1e-4)
            # The default value is method specific. For example for Nelder-Mead they are xatol=1e-4, fatol=1e-4

            if cost_vs_iteration != []:
                ### Plot of iterations. 
                ### Works only if you chose the cost_func that plots iteration.
                plt.figure(figsize=(7, 4))
                plt.rcParams['font.size'] = 13
                plt.plot(cost_vs_iteration)
                plt.xlabel("Iteration")
                plt.ylabel("Cost")
                plt.show()
            
            ### Select the optimal parameters (betas,gammas) found.
            betas = list(res.x[:p])
            gammas = list(res.x[p:])
            print(f"Final parameters (after minimization): betas, gammas = {betas}, {gammas}")

            ### Minimum cost (energy) reached with minimization.
            E_min = res.fun + constant
            print(f"E_min = res.fun + constant = {E_min}")
            # E_min = res.fun -A -B  
            # print(f"E_min = res.fun - A - B = {E_min}")
            
            DATA_FILE.write(f"\nE_min = {E_min}")
            DATA_FILE.write(f'\nE_min\'s parameters: betas = {betas}, gammas = {gammas}\n')

            ### Update the lowest energy solution index "i".
            if E_min < E_best:
                    E_best = E_min
                    i_best = i 
                    print("***UPDATING THE BEST ENERGY***\n")
            else:
                print("***NOT UPDATING***\n")
        
                    
            # ****************************  RUN THE CIRCUIT  ******************************
            ### Assign to the ansatz the 2p parameters found, then run the circuit.
            pars = betas + gammas
            qc = ansatz.assign_parameters(pars)
            qc.measure_all()
            qc_isa = pm.run(qc)
            result = sampler.run([qc_isa], shots=1024).result()
            samp_dist = result[0].data.meas.get_counts()
       
            
            # ****************************  POST PROCESS  **********************************
            # Create a dataframe out of the sampling's results.
            df = pd.DataFrame(samp_dist.items()).rename(columns={0: 'states', 1: 'counts'})
            _, _, NUM_ANC, _ = get_circuit_parameters(subsets, verbose=False)
            df['states'] = df['states'].apply(lambda x: x[NUM_ANC:]) # remove ancillary bits
            df = df.groupby(['states']).sum()
            
            # Create a dictionary with states and occurrences.
            d = df['counts'].to_dict()
            lists = sorted(d.items(), key=lambda item: item[1], reverse=True)
        
            # Invert bit order ("01101 -> 10110")      
            d = invert_counts(d)
        
            
            # **************************** PLOT THE i-TH DATAFRAME ****************************
            # plt.figure(figsize=(15,5))
            # plt.rcParams['font.size'] = 13
            # plt.title(f"Ansatz with p={p}, {i} of {random_attempts}")
            # plt.ylim(0,1024)
            # plt.grid()
            df = pd.DataFrame(d.items())
            df = df.rename(columns={0: 'states', 1: f'counts_p{p}_{i}of{random_attempts}'})    
            df = df.sort_values(f'counts_p{p}_{i}of{random_attempts}', ascending=False) 
            # plot_histogram_of_df_column(df, f'counts_p{p}_{i}of{random_attempts}', EXACT_COVERS, init_name, title='')
            # ax = sns.barplot(x='states', y=f'counts_p{p}_{i}of{random_attempts}', data=df)
            
            # # Make labels with percentages.
            # labels = df[f'counts_p{p}_{i}of{random_attempts}'].apply(lambda x: (x/df[f'counts_p{p}_{i}of{random_attempts}'].sum())*100).round(1).astype('str') + '%'
            # for container in ax.containers:
            #     ax.bar_label(container, labels=labels, fontsize=11)
                
            # # Highlight with red the exact covers
            # highlight_correct_ticks(ax, EXACT_COVERS)
            # plt.show()
    
            # Merge dataframes.
            if i == 1:
                df_final = df
            else:
                df_final = pd.merge(df_final, df, on="states", how="outer")    
    
            # if df['states'].iloc[0] == MEC:    
            #     DATA_FILE.write("\n### Most frequent state is MEC ###\n")
            #     counter += 1
        
        # ################### SAVE TO CSV AND FIX BETA,GAMMA ###################
        # Save to csv.
        df_final.to_csv(FILENAME, index=False)
        title = FILENAME +"\n"+ init_string
        plot_histogram_of_best_column(df_final, f'counts_p{p}_{i_best}of{random_attempts}', EXACT_COVERS, init_name, title=title)
        
        ######################################################################
        DATA_FILE.write("\n*******************************")
        DATA_FILE.write(f"\nAttempt that reached the best result with E_min = {E_best} is #{i_best} ")
        DATA_FILE.write(f"\nMost frequent state was MEC / random attempts = {counter} / {random_attempts} = {round((counter/random_attempts)*100, 1)}%\n")
        
    
        print(f"\nTOTAL ELAPSED TIME: {(time.time() - TOTAL_start_time)/60} minutes.")
        DATA_FILE.write(f"\nTOTAL ELAPSED TIME: {(time.time() - TOTAL_start_time)/60} minutes.\n")


# In[ ]:


print("ciao")


# ## Read files.

# #### Stampa UN SOLO file che contiene una sottostringa.

# In[4]:


# Find the files that contain a substring in a certain path.
substrings = ["all1", "mail3"]
path = "./FILES_PER-IMMAGINI-LATEX/tutte-le-istanze_all0_all1_20ra/"
FILENAME_list, DATA_FILENAME_list = find_files_containing_string(substrings, path)


# Select only the first file.
FILENAME = FILENAME_list[0]
DATA_FILENAME = DATA_FILENAME_list[0]


# Plot.
plot_file(FILENAME, DATA_FILENAME, colorchosen='indigo', alpha=0.4,
          dont_show_in_title = [], 
          figsize=(8, 3), dpi=200, N=10)


# #### Stampa DIVERSI file che contengono una sottostringa

# In[30]:


# Find files that contain a substring in a certain path.
substrings = ["all1"]
path = "./FILES_PER-IMMAGINI-LATEX/tutte-le-istanze_all0_all1_20ra/"
# path = "./FILES_PER-IMMAGINI-LATEX/varie_k_tol1e4_all1/"
FILENAME_list, DATA_FILENAME_list = find_files_containing_string(substrings, path)

# Plot
plot_list_of_files(FILENAME_list, DATA_FILENAME_list, colorchosen="steelblue", alpha=0.6,
                   init_name="all1",
                   dont_show_in_title=["i", "k"], 
                   dont_show_in_titles=["n", "p", "ra", "k", "init"], 
                   figsize=(10, 13), dpi=300, N=12)


# ### funzione che stampa un subplot per ogni istanza. in ogni subplot mette piu barplot di colore diverso in base a  "order"

# ### Plot  di all0 vs all1 con random parameters

# In[3]:


multiple_barplot(path = "./FILES_PER-IMMAGINI-LATEX/tutte-le-istanze_all0_all1_20ra/",
                 string_order = [ 'all1', 'all0'],
                 order_labels = ['$all1$', '$all0$'],
                 color_map = {"$|0\\rangle$ initialization": 'crimson', "$|1\\rangle$ initialization": 'steelblue'},
                 filename_to_save = "all0_all1_random.pdf",
                 show_title=False)


# ### Plot di confronto tra varie k

# In[135]:


multiple_barplot(path = "./FILES_PER-IMMAGINI-LATEX/varie_k_options_tol1e4_all1/",
                 string_order = ['L=n', 'maxLEC', 'LMEC'],
                 order_labels = ['$L=n$', '$L=$max($L_{EC}$)', '$L=L_{MEC}$'],
                 color_map = {'$L=n$': 'darkorange', '$L=$max($L_{EC}$)': 'crimson', '$L=L_{MEC}$': 'steelblue'},
                 filename_to_save = "performance_k.pdf",
                 show_title = False)

