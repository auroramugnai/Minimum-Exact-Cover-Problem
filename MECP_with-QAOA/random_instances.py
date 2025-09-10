info_dim6 = {1: {'exact_covers': ['010101', '001100'],
                 'mec': '001100',
                 'subsets': [{1, 3, 4, 5, 6, 7, 8, 9, 11, 12},
                             {2, 11, 4, 5},
                             {1, 2, 4, 5, 6, 11},
                             {3, 7, 8, 9, 10, 12},
                             {2, 3, 4, 6, 7, 8, 9, 10, 11, 12},
                             {1, 6}]},
             2: {'exact_covers': ['101110', '100011'],
                 'mec': '100011',
                 'subsets': [{8, 9, 11},
                             {2, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                             {1, 10, 12, 7},
                             {2, 3},
                             {4, 5, 6},
                             {1, 2, 3, 7, 10, 12}]},
             3: {'exact_covers': ['001111', '101001'],
                 'mec': '101001',
                 'subsets': [{2, 10, 4, 5},
                             {1, 3, 4, 5, 6, 7, 9, 10, 11, 12},
                             {3, 7, 8, 9, 11, 12},
                             {2, 10},
                             {4, 5},
                             {1, 6}]},
             4: {'exact_covers': ['110001', '001001'],
                 'mec': '001001',
                 'subsets': [{3, 4, 12, 7},
                             {5, 6},
                             {3, 4, 5, 6, 7, 12},
                             {1, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 11},
                             {1, 2, 8, 9, 10, 11}]},
             5: {'exact_covers': ['101110', '001101'],
                 'mec': '001101',
                 'subsets': [{5, 6},
                             {1, 2, 4, 5, 7, 8, 9, 10, 11, 12},
                             {1, 2, 3, 4, 7, 8},
                             {9, 11},
                             {10, 12},
                             {10, 12, 5, 6}]},
             6: {'exact_covers': ['010011', '001001'],
                 'mec': '001001',
                 'subsets': [{1, 2, 4, 5, 7, 8, 9, 10, 11, 12},
                             {1, 3, 6},
                             {1, 3, 6, 7, 10, 11},
                             {1, 2, 3, 4, 5, 7, 9, 10, 11, 12},
                             {10, 11, 7},
                             {2, 4, 5, 8, 9, 12}]},
             7: {'exact_covers': ['001111', '101010'],
                 'mec': '101010',
                 'subsets': [{1, 3, 8, 9, 10, 12},
                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 12},
                             {2, 11, 6, 7},
                             {9, 10},
                             {4, 5},
                             {8, 1, 3, 12}]},
             8: {'exact_covers': ['101001', '011000'],
                 'mec': '011000',
                 'subsets': [{2, 4, 7, 11, 12},
                             {1, 2, 4, 5, 6, 7, 10, 11, 12},
                             {8, 9, 3},
                             {1, 3, 4, 5, 6, 8, 9, 10, 11, 12},
                             {1, 2, 3, 4, 5, 6, 7, 10, 11, 12},
                             {1, 10, 5, 6}]},
             9: {'exact_covers': ['101101', '101010'],
                 'mec': '101010',
                 'subsets': [{8, 9, 10, 11},
                             {1, 2, 3, 4, 5, 6, 9, 10, 11, 12},
                             {1, 3, 5},
                             {2, 12},
                             {2, 4, 6, 7, 12},
                             {4, 6, 7}]},
             10: {'exact_covers': ['001101', '010100'],
                 'mec': '010100',
                 'subsets': [{1, 2, 4, 6, 7, 8, 9, 10, 11, 12},
                             {1, 2, 5, 6, 9, 10, 12},
                             {1, 2, 10},
                             {3, 4, 7, 8, 11},
                             {1, 2, 4, 5, 6, 7, 8, 9, 10, 12},
                             {9, 12, 5, 6}]},
             'U': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}



def compute_mean_valency_variance(info_dim6, verbose=False):

    mean_valency_values = []
    for key, data in info_dim6.items():
        if key == 'U':  # skip the universe set
            continue
        subsets = data['subsets']
        
        mean_valency = compute_mean_valency(subsets, verbose=verbose)
        mean_valency_values.append(mean_valency)
        
    variance = np.var(mean_valency_values)
    
    if verbose:
        print(f"Variance = {variance}")
    return variance


############################################################################################
def compute_mean_valency(subsets, verbose=False):
    """
    Calculates the mean valency for a list of subsets.

    Parameters
    ----------
    subsets : list of set
        A list where each element is a set representing a subset.

    Returns
    -------
    mean_valency : float
        The average number of other subsets each subset intersects with.
    """

    # Find the list of intersections:
    # For each subset, find the subsets it has a non-zero intersection with.
    list_of_intersections = find_intersections_lists(subsets)

    # Calculate the valency for each subset (i.e., the number of subsets it intersects with).
    valencies = [len(x) for x in list_of_intersections]
    mean_valency = np.mean(valencies)
    if verbose:
        print("Valencies: ", valencies)
        print("Mean valency: ", mean_valency)
    
    return mean_valency


############################################################################################
def fix_small_sets(subsets, min_size=2, protected_positions=None):
    """
    Ensures that each subset in the list has at least `min_size` elements,
    except for those at specified protected positions.

    Args:
        subsets (list[set]): A list of subsets (Python sets).
        min_size (int, optional): The minimum allowed size of a subset.
                                  Defaults to 2.
        protected_positions (list[int], optional): Indices of subsets that
                                  must remain unchanged. Defaults to None.

    Notes:
        - The function modifies `subsets` in place.
        - Protected subsets (in `protected_positions`) are never altered.
        - If a subset has fewer than `min_size` elements, the function
          attempts to transfer elements from non-protected donor subsets
          that have more than `min_size` elements.
        - The repair process continues iteratively until no further
          changes are possible or all subsets satisfy the size constraint.
    """
    if protected_positions is None:
        protected_positions = []

    changed = True
    while changed:
        changed = False
        for i, s in enumerate(subsets):
            if i in protected_positions:
                continue  # Skip protected subsets

            if len(s) < min_size:
                # Find candidate donors: subsets that are not protected
                # and have more than min_size elements
                donors = [
                    j for j, other in enumerate(subsets)
                    if j != i and j not in protected_positions and len(other) > min_size
                ]
                if not donors:
                    continue  # No valid donor available

                # Randomly choose a donor subset and transfer one element
                donor = random.choice(donors)
                elem = random.choice(list(subsets[donor]))
                subsets[donor].remove(elem)
                subsets[i].add(elem)
                changed = True


############################################################################################                
def repair_subsets(subsets, U, mec_position):
    """
    Repairs a collection of subsets so that they satisfy the constraints 
    of the Minimum Exact Cover Problem (MEC).

    Args:
        subsets (list[set]): List of subsets (Python sets) to be repaired.
        U (set): The universe of elements that must be fully covered 
                 by the union of subsets.
        mec_position (list[int]): Indices of subsets that are part of 
                 the exact cover solution and must remain unchanged.

    The function enforces the following constraints:
        - The union of all subsets must equal U.
        - Duplicate elements are removed so that each element 
          appears in exactly one subset.
        - No subset (except those in mec_position) can have fewer than 2 elements.
        - Subsets in mec_position are preserved as-is.

    Returns:
        list[set]: The repaired list of subsets.

    Raises:
        ValueError: If no valid subset can be found for a duplicated element.

    Note:
        The "mean valency" constraint is not implemented.
    """
    new_subsets = copy.deepcopy(subsets)

    # --- Step 1: Add missing elements ---
    current_union = set.union(*new_subsets)
    missing = U - current_union
    for elem in missing:
        # Place the missing element in a subset that is not protected (not in mec_position)
        candidates = [s for i, s in enumerate(new_subsets) if i not in mec_position]
        random.choice(candidates).add(elem)

    # --- Step 2: Remove duplicates ---
    # Build a mapping: element -> subsets containing that element
    elem_to_subsets = {}
    for i, subset in enumerate(new_subsets):
        for elem in subset:
            elem_to_subsets.setdefault(elem, []).append(i)

    # Keep only elements that appear in more than one subset
    elem_to_subsets = {k: v for k, v in elem_to_subsets.items() if len(v) > 1}

    for elem, subsets_that_have_elem in elem_to_subsets.items():
        # Select one subset to keep the element
        # Priority: protected subsets, then subsets with fewer than 3 elements, otherwise any
        candidates = [i for i in subsets_that_have_elem if i in mec_position]
        if not candidates:
            candidates = [i for i in subsets_that_have_elem if len(new_subsets[i]) < 3]
        if not candidates:
            candidates = subsets_that_have_elem

        if not candidates:
            raise ValueError(f"No valid subset found for element {elem}")

        keep = random.choice(candidates)

        # Remove the element from all other subsets
        for subset_idx in subsets_that_have_elem:
            if subset_idx != keep:
                new_subsets[subset_idx].discard(elem)

    # --- Step 3: Ensure minimum subset size ---
    # Fix subsets with fewer than 2 elements (except protected ones)
    fix_small_sets(new_subsets, protected_positions=mec_position)

    return new_subsets


############################################################################################
def build_mec(U, mec_len):
    """
    Build 'mec_len' disjoint subsets that partition U.
    """
    u = list(U)
    random.shuffle(u)
    
    mec = [[] for _ in range(mec_len)]
    for elem in u:
        mec[random.randrange(mec_len)].append(elem)
    
    mec = [set(s) for s in mec]# Converto ogni gruppo in un set

    # se un gruppo è vuoto, riassegno un elemento da un altro gruppo
    for i, s in enumerate(mec):
        if len(s) < 2:
            for j, other in enumerate(mec):
                set_is_empty = i != j and len(s) == 0 and len(other) > 1
                set_has_one_elem = i != j and len(s) == 1 and len(other) > 2
                if set_is_empty or set_has_one_elem:
                    elem = other.pop()
                    s.add(elem)
                    break
    return mec


############################################################################################
def build_not_mec(U, mec_len, n, min_w_non_mec):
    """
    Build 'n - mec_len' subsets that are not part of the mec.
    Each subset has at least 'min_w_non_mec' elements.
    The mean valency of the final collection of subsets (mec + not_mec)
    is between 2 and 5."""
    CREATE_NEW_SUBSETS = True
    while CREATE_NEW_SUBSETS == True:
        not_mec = []
        for _ in range(n - mec_len):
            random_two_elem_set = set(random.sample(sorted(U), min_w_non_mec))
            not_mec.append(random_two_elem_set)

        ## Shuffle the 'not_mec' part of subsets
        subsets_tmp  = mec + random.sample(not_mec, len(not_mec))
        
        ## Add elements to the 'not_mec' subsets until the mean valency is between 2 and 5.
        mean_valency_ctrl = True
        for i in range(elems_to_be_added):
            for s in subsets_tmp[mec_len::]: # exlcuding the mec
                not_yet = list(U - s) # elements 1-12 not in 's' yet.
                s.add(random.choice(not_yet))
        
                updated_mean_valency = compute_mean_valency(subsets_tmp)
                if (2 < updated_mean_valency < 5):
                    continue # Go to the next subset.
                else:
                    mean_valency_ctrl = False # Create subsets again.
                    break
            if mean_valency_ctrl == False: # Create subsets again.
                break

            ## Check if we've added all required elements
            if i == elems_to_be_added - 1 and mean_valency_ctrl==True:
                # The computation has ended successfully.
                # All subsets have been generated with the required mean valency.
                CREATE_NEW_SUBSETS = False
                break
        
    return subsets_tmp


############################################################################################
def choose_ec_subsets(mec, subsets):
    while True:
        ec_length = random.randint(len(mec)+1, 4)  # how many subsets to select
        ec = random.sample(subsets, ec_length)

        # Constraint: do not select both the first two substets (the mec)
        if not all(mec_subset in ec for mec_subset in mec):
            break
    
    # Sort as they appear in 'subsets'
    ec = sorted(ec, key=lambda x: subsets.index(x))
    positions = [i for i,s in enumerate(subsets) if s in ec]

    return ec, positions


############################################################################################
def plot_mv_vs_instance(variance_attempt, num_variance_attempts, instance_dict):
        
    ## Plot.
    if variance_attempt == 0:
        plt.figure()

    # Compute the mean valency values for each instance.
    mv_list = []
    for k,v in instance_dict.items():
        if k == 'U':
            continue
        mv_list.append(compute_mean_valency(v['subsets']))

    # markersize gets littler as the number of attempts increases.
    markersize = max(15 - variance_attempt*3, 2)

    # fix a color for each attempt.
    c = plt.cm.viridis(1-variance_attempt / num_variance_attempts)

    # Plot the mean valency values for this attempt.
    plt.plot(mv_list, 'o', color =c, markersize=markersize, markerfacecolor='None',
             label=f'Attempt {variance_attempt}, Variance = {my_variance:.4f}')

    # Plot the mean of the mean_valency values for this attempt.
    plt.axhline(np.mean(mv_list), color =c, label=f'Mean valency average attempt {variance_attempt}')

    # show variance of each attempt as a vertical segment.
    plt.errorbar([len(mv_list)+0.+0.1*variance_attempt], [np.mean(mv_list)], 
                 yerr=[[my_variance], [my_variance]], 
                 fmt='o', color=c, capsize=10, label='My variance')  

    if variance_attempt == num_variance_attempts - 1:
        plt.xlabel("Instance index")
        plt.ylabel("Mean valency")
        plt.legend(loc='best')
        plt.grid()
        plt.show()
                
                
########################################################################################
########################################################################################
########################################################################################
if __name__ == "__main__":
    import copy
    import random
    import pprint
    import numpy as np
    import matplotlib.pyplot as plt
    
    from utils_to_study_an_instance import find_intersections_lists, build_instance_graph

    
    # Fixed data.
    U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    n = 6
    
    max_w_non_mec = 10
    min_w_non_mec = 2
    elems_to_be_added = max_w_non_mec - min_w_non_mec
    

    NUM_INSTANCES = 10 # number of instances to be generated
    instance_dict = {} 
    instance_dict['U'] = U

    from Wang_instances import info_dim6 as info_dim6_wang
    wang_variance = compute_mean_valency_variance(info_dim6_wang)
    
    my_variance_list = []
    num_variance_attempts = 10
    num_ec_repair_attempts = 100
    my_variance_old = -1

    for variance_attempt in range(num_variance_attempts): 
        print(f"\r--- Variance attempt {variance_attempt}/{num_variance_attempts} ---", end="")
        
        for instance_idx in range(NUM_INSTANCES):
            # print(f"\n********************************\n--- Building instance {instance_idx} ---\n********************************\n")
            START_INSTANCE_AGAIN = True
            CONDITION_MET = False
            while START_INSTANCE_AGAIN == True:
                # print(f"\n--- Restarting instance {instance_idx} ---")
                ## Generate 'mec_len' disjoint subsets that partition U
                mec_len = random.randint(2,3)
                mec = build_mec(U, mec_len)

                subsets = build_not_mec(U, mec_len, n, min_w_non_mec)
                ec_tmp, positions = choose_ec_subsets(mec, subsets)

                mec_position = [i for i,s in enumerate(ec_tmp) if s in mec]
                

                ## Try to repair ec_tmp in a way that makes 2 < mv < 5.
                for _ in range(num_ec_repair_attempts):
                    # print(f"\rec repair attempt {_}/{num_ec_repair_attempts} ---", end="")
                    ec_tmp = repair_subsets(ec_tmp, U, mec_position)
                    subsets_tmp = subsets
                    for i,subset in zip(positions, ec_tmp):
                        subsets_tmp[i] = subset
                    
                    if not(all(len(s) >= 2 for s in subsets_tmp)):
                        CONDITION_MET = False
                        continue

                    mv = compute_mean_valency(subsets_tmp)
                    if 2 < mv < 5:
                        CONDITION_MET = True # success :)
                        ec = ec_tmp
                        subsets = subsets_tmp
                        # print("Condition met!")
                        break
                    else:
                        CONDITION_MET = False

                
                if CONDITION_MET:
                    ec = ec_tmp
                    subsets = subsets_tmp

                    ec_str = ''.join(['1' if s in ec else '0' for s in subsets])
                    mec_str = ''.join(['1' if s in mec else '0' for s in subsets])

                    # Shuffle subsets and accordingly ec and mec strings.
                    shuffled_indices = list(range(len(subsets)))
                    random.shuffle(shuffled_indices)
                    subsets = [subsets[i] for i in shuffled_indices]
                    ec_str = ''.join([ec_str[i] for i in shuffled_indices])
                    mec_str = ''.join([mec_str[i] for i in shuffled_indices])

                    instance_dict[instance_idx] = {
                        'exact_covers': [ec_str, mec_str],
                        'mec': mec_str,
                        'subsets': list(subsets)
                    }
                    
                    # Compute the variance of mean valency for the current instance
                    my_variance = compute_mean_valency_variance(instance_dict)
                    if my_variance < my_variance_old:
                        START_INSTANCE_AGAIN = True # Discard the instance and restart
                    else:
                        my_variance_old = my_variance # Update      
                        START_INSTANCE_AGAIN = False

                                            
        ####################################################################################
        ## Here, the dictionary 'instance_dict' has been filled with NUM_INSTANCES instances.
        ## The dictionary has 'my_variance' variance.
        my_variance_list.append(my_variance)
        print("my variance:", my_variance)
        
        plot_mv_vs_instance(variance_attempt, num_variance_attempts, instance_dict)
       

    pprint.pprint(instance_dict)
    print("my variance list:", my_variance_list)

    # ## Save the instance_dict if my_variance > 0.48.
    # import pickle
    # if my_variance > 0.48:
    #     with open("instance_dict.pkl", "wb") as f:
    #         pickle.dump(instance_dict, f)
    #     print("instance_dict salvato in instance_dict.pkl")
    # else:
    #     print("instance_dict NON salvato.")
    # import pickle
    # with open("instance_dict.pkl", "rb") as f:
    #     instance_dict = pickle.load(f)
    

    ## Plot the variance values.
    plt.figure()
    plt.errorbar(np.linspace(0, num_variance_attempts, num_variance_attempts), my_variance_list, fmt='o')
    plt.axhline(wang_variance, color='r', label='Wang variance') 
    plt.grid()
    plt.ylabel("Variance")
    plt.xlabel("Attempt index")
    plt.legend(loc='best')
    plt.show()
