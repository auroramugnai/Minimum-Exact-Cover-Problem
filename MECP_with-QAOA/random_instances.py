info_dim6 = {
 1: {'exact_covers': ['101110', '010001'],
     'mec': '010001',
     'subsets': [{9, 6}, {9, 10, 6, 1}, {5, 12}, {2, 3, 4, 10}, 
                 {1, 7, 8, 11}, {2, 3, 4, 5, 7, 8, 11, 12}]},
 2: {'exact_covers': ['010011', '001100'],
     'mec': '001100',
     'subsets': [{12, 5, 6, 7}, {1, 5, 7, 8, 9, 12}, {1, 2, 3, 5, 8, 9, 10, 11, 12}, 
                 {4, 6, 7}, {4, 6}, {2, 3, 10, 11}]},
 3: {'exact_covers': ['110011', '001100'],
     'mec': '001100',
     'subsets': [{8, 3, 6}, {1, 4, 5, 12}, {1, 11}, {2, 3, 4, 5, 6, 7, 8, 9, 10, 12}, 
                 {11, 10, 7}, {2, 9}]},
 4: {'exact_covers': ['011010', '000101'],
     'mec': '000101',
     'subsets': [{2, 11, 4, 5}, {8, 10, 12}, {9, 4, 5, 6}, {1, 2, 3, 4, 5, 7, 8, 9, 10, 11}, 
                 {1, 2, 3, 7, 11}, {12, 6}]},
 'U': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}


def compute_mean_valency_variance(info_dim6, verbose=False):

    mean_valency_values = []
    for key, data in info_dim6.items():
        if key == 'U':  # skip the universe set
            continue
        subsets = data['subsets']
        
        mean_valency_values.append(compute_mean_valency(subsets, verbose=verbose))
        
    variance = np.var(mean_valency_values)
    
    if verbose:
        print(f"Variance = {variance}")
    return variance


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



import random

def fix_small_sets(subsets, min_size=2, protected_positions=None):
    """
    Ensures that every subset in the provided list has at least `min_size` elements, except for those at specified protected positions.

    Parameters
    ----------
        subsets: A list of sets, where each set represents a subset of elements.
        min_size: The minimum number of elements required in each subset (default is 2).
        protected_positions: List of indices indicating subsets that must not be modified.

    Notes
    -----
    - The function modifies `subsets` in place.
    - Subsets at indices specified in `protected_positions` are never changed.
    - If a subset has fewer than `min_size` elements, the function attempts to add elements from other non-protected subsets that have more than `min_size` elements.
    - The process repeats until no further changes are possible or all subsets meet the minimum size requirement.
    """

    if protected_positions is None:
        protected_positions = []

    changed = True
    while changed:
        changed = False
        for i, s in enumerate(subsets):
            if i in protected_positions:
                continue  # non toccare questo subset

            if len(s) < min_size:
                # Trova donatori validi (non protetti e con più di min_size elementi)
                donors = [
                    j for j, other in enumerate(subsets)
                    if j != i and j not in protected_positions and len(other) > min_size
                ]
                if not donors:
                    continue  # nessun donatore disponibile

                donor = random.choice(donors)
                elem = random.choice(list(subsets[donor]))
                subsets[donor].remove(elem)
                subsets[i].add(elem)
                changed = True


def repair_subsets(subsets, U, mec_position):
    """
    Repairs a collection of subsets to satisfy specific constraints for the Minimum Exact Cover Problem.
    Args:
        subsets (list of set): List of subsets (as sets) to be repaired.
        U (set): The universe of elements that must be covered by the union of subsets.
        mec_position (list of int): Indices of subsets that are part of the minimum exact cover and must not be modified.
    The function attempts to:
        - Ensure the union of all subsets covers U.
        - Remove duplicate elements across subsets (each element appears in only one subset).
        - Ensure no subset (except those in mec_position) has fewer than 2 elements.
        - Subsets in mec_position are not modified.
    Returns:
        list of set: The repaired list of subsets.
    Raises:
        RuntimeError: If the repair process fails to satisfy the constraints.
        ValueError: If no valid subset can be found for a duplicate element.
    Note:
        The mean valency constraint is not implemented.
    """
    new_subsets = copy.deepcopy(subsets)

    # Add missing elements
    # print("... Add missing elements ...")
    current_union = set.union(*new_subsets)
    missing = U - current_union
    for elem in missing:
        # Choose a subset not in mec_position
        candidates = [s for i, s in enumerate(new_subsets) if i not in mec_position]
        random.choice(candidates).add(elem)


    # Remove duplicates
    # print("... Removing duplicates ...")
    elem_to_subsets = {}
    for i, subset in enumerate(new_subsets):
        for elem in subset:
            elem_to_subsets.setdefault(elem, []).append(i)

    # Rimuovi le voci con lista di 0 o 1 elemento
    elem_to_subsets = {k: v for k, v in elem_to_subsets.items() if len(v) > 1}
    
    for elem, subsets_that_have_elem in elem_to_subsets.items():
        # Choose the one to keep
        # Exclude the mec subset and those subsets that have length <=2
        # print("[i for i in subsets_that_have_elem]", [i for i in subsets_that_have_elem])
        candidates = [i for i in subsets_that_have_elem if i in mec_position]
        if candidates == []:
            candidates = [i for i in subsets_that_have_elem if len(ec[i]) < 3]
            
        if candidates == []:
            candidates = subsets_that_have_elem
            
        if not candidates:
            raise ValueError(f"Nessun subset valido trovato per l'elemento {elem}")
        keep = random.choice(candidates)

        for subset_idx in subsets_that_have_elem:
            if subset_idx != keep:
                new_subsets[subset_idx].discard(elem)

    # print(f"    EC_tmp = {new_subsets}")

    # Ensure no subset is smaller than 2 elements
    # print("... Fixing small sets ...")
    fix_small_sets(new_subsets, protected_positions=mec_position)
    # print("    EC_tmp = ", new_subsets)

    return new_subsets

  
if __name__ == "__main__":
    import copy
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    from utils_to_study_an_instance import find_intersections_lists, build_instance_graph

    
    # Fixed data.
    U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    n = 6
    
    max_w = 5 # maximum weight of mec subsets
    min_w = 4 # minimum weight of mec subsets
    max_w_non_mec = 7 # keep them small
    min_w_non_mec = 2
    elems_to_be_added = max_w_non_mec - min_w_non_mec
    

    NUM_INSTANCES = 10 # number of instances to be generated
    instance_dict = {} 
    instance_dict['U'] = U

    from Wang_instances import info_dim6 as info_dim6_wang
    wang_variance = compute_mean_valency_variance(info_dim6_wang)
    
    variance_attempt = 0
    VARIANCE_IS_NOT_SIMILAR_TO_WANG = True
    while VARIANCE_IS_NOT_SIMILAR_TO_WANG == True and variance_attempt < 100000: 
        print(f"\r--- Variance attempt {variance_attempt}/100000 ---", end="")
        variance_attempt += 1
        for instance_idx in range(NUM_INSTANCES):
            # print(f"\n--- Building instance {instance_idx} ---")
            START_INSTANCE_AGAIN = True
            CREATE_NEW_SUBSETS = True

            discard = 0
            while START_INSTANCE_AGAIN == True:
                # Generate the MEC with 2 subsets.
                subsets = []
                mec = []
                mec_len = random.randint(2,3)

                # Generate 'mec_len' disjoint subsets that partition U
                u = list(sorted(U))
                random.shuffle(u)
                ws_mec = [random.randint(min_w, max_w) for _ in range(mec_len - 1)]
                # Ensure total size does not exceed len(U)
                ws_mec.append(len(U) - sum(ws_mec))
                start = 0
                for w in ws_mec:
                    subset = set(u[start:start + w])
                    subsets.append(subset)
                    mec.append(subset)
                    start += w
                # print(f"    MEC = {mec}")



                while CREATE_NEW_SUBSETS == True:

                    # Generate the other subsets.
                    not_mec = []
                    for _ in range(n - mec_len):
                        # For now, each subset has only min_w_non_mec=2 elements.
                        random_two_elem_set = set(random.sample(sorted(U), min_w_non_mec))
                        not_mec.append(random_two_elem_set)


                    # Shuffle the 'not_mec' part of subsets,
                    subsets = mec + random.sample(not_mec, len(not_mec))
                    subsets_tmp = copy.deepcopy(subsets)

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

                        # Check if we've added all required elements
                        if i == elems_to_be_added - 1 and mean_valency_ctrl==True:
                            # The computation has ended successfully.
                            CREATE_NEW_SUBSETS = False
                            break
                
                # All subsets have been generated with the required mean valency.
                subsets = subsets_tmp

                # Now we generate an EC by selecting 3 or 4 subsets (not both the 1st and the 2nd)
                while True:
                    ec_length = random.randint(mec_len+1, 6)  # how many subsets to select
                    ec = random.sample(list(subsets), ec_length)
                    
                    # Sort as they appear in subsets
                    ec = sorted(ec, key=lambda x: list(subsets).index(x))
                    positions = [i for i,s in enumerate(subsets) if s in ec]
                    
                    # Constraint: do not select both the first two elements
                    if not all(mec_subset in ec for mec_subset in mec):
                        break
                
                # print(f"\n    EC_tmp = {ec}")
                mec_position = [i for i,s in enumerate(ec) if s in mec]
                
                max_attempts = 1000
                for attempt in range(max_attempts):
                    ec = repair_subsets(ec, U, mec_position)
                    subsets_tmp = np.array(copy.deepcopy(subsets))
                    for i,subset in zip(positions,ec):
                        subsets_tmp[i] = subset
                    
                    # Condition on global mean valency
                    mv = compute_mean_valency(subsets_tmp)
                    # print(f"Attempt {attempt}: mean_valency = {mv}\n")
                
                    if 2 < mv < 5 and all(len(s) >= 2 for s in subsets_tmp):
                        # print("SUCCESS: breaking...")
                        START_INSTANCE_AGAIN = False
                        break
                    else:
                        # If no valid configuration found after all attempts
                        # print(f"Could not repair subsets in {max_attempts} attempts")
                        break
                    
                if  CREATE_NEW_SUBSETS == False:
                    # print(f"\n--> Attempt {attempt} repaired EC.")
                    # print(f"subsets = {subsets_tmp}")
                    # print(f"mean_valency = {mv}")

                    ec_str = ''.join(['1' if s in ec else '0' for s in subsets_tmp])
                    mec_str = ''.join(['1' if s in mec else '0' for s in subsets_tmp])

                    # print("ec_str =", ec_str)
                    # print("mec_str =", mec_str)


                    instance_dict[instance_idx] = {
                        'exact_covers': [ec_str, mec_str],
                        'mec': mec_str,
                        'subsets': list(subsets_tmp)
                    }

                    # Compute the variance of mean valency for the current instance
                    my_variance = compute_mean_valency_variance(instance_dict)

                    # For the first instance, initialize the old variance
                    if instance_idx == 0:
                        my_variance_old = my_variance
                    else:
                        if my_variance < my_variance_old:
                            # Discard the whole instance and restart
                            discard += 1
                            START_INSTANCE_AGAIN = True

                        # Update the old variance for the next comparison
                        my_variance_old = my_variance


        if my_variance > wang_variance - 0.15 :
            # End the computation
            VARIANCE_IS_NOT_SIMILAR_TO_WANG = False
        
    
    print(f"\n {discard} instances discarded")
    print(f"my_variance = {my_variance}")
    print(f"wang_variance = {wang_variance}")
    
    import pickle

    if my_variance > 0.48:
        with open("instance_dict.pkl", "wb") as f:
            pickle.dump(instance_dict, f)
        print("instance_dict salvato in instance_dict.pkl")
    else:
        print("instance_dict NON salvato (my_variance <= 0.48)")



# with open("instance_dict.pkl", "rb") as f:
#     instance_dict = pickle.load(f)

# print(instance_dict)