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
                ## Generate 'mec_len' disjoint subsets that partition U
                subsets = []
                mec = []
                mec_len = random.randint(2,3)
                
                ## Randomly choose the weights of the mec subsets, between min_w and max_w.
                ws_mec = [random.randint(min_w, max_w) for _ in range(mec_len - 1)]
                ws_mec.append(len(U) - sum(ws_mec))

                u = list(sorted(U))
                random.shuffle(u)
                start = 0
                for w in ws_mec:
                    subset = set(u[start:start + w])
                    subsets.append(subset)
                    mec.append(subset)
                    start += w
                # print(f"    MEC = {mec}")

                ## Generate the other subsets ensuring that
                ## the global mean valency is between 2 and 5.
                while CREATE_NEW_SUBSETS == True:
                    not_mec = []
                    for _ in range(n - mec_len):
                        # For now, each subset has only min_w_non_mec=2 elements.
                        random_two_elem_set = set(random.sample(sorted(U), min_w_non_mec))
                        not_mec.append(random_two_elem_set)

                    ## Shuffle the 'not_mec' part of subsets,
                    subsets = mec + random.sample(not_mec, len(not_mec))
                    subsets_tmp = copy.deepcopy(subsets)

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
                            CREATE_NEW_SUBSETS = False
                            break
                
                ## All subsets have been generated with the required mean valency.
                subsets = subsets_tmp

                ## Generate an EC by selecting a random number of subsets 
                ## (not both the 1st and the 2nd -- the mec)
                while True:
                    ec_length = random.randint(mec_len+1, 6)  # how many subsets to select
                    ec = random.sample(list(subsets), ec_length)
                    
                    # Sort as they appear in 'subsets'
                    ec = sorted(ec, key=lambda x: list(subsets).index(x))
                    positions = [i for i,s in enumerate(subsets) if s in ec]
                    
                    # Constraint: do not select both the first two substets (the mec)
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
                        # Exit
                        break
                    else:
                        # If no valid configuration found after all attempts
                        # print(f"Could not repair subsets in {max_attempts} attempts")
                        # Exit (with error)
                        break
                    
                if  START_INSTANCE_AGAIN == False:
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