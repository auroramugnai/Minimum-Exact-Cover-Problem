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


def compute_mean_valency_variance(info_dim6):
    
    mean_valency_values = []
    for key, data in info_dim6.items():
        if key == 'U':  # skip the universe set
            continue
        subsets = data['subsets']
        
        mean_valency_values.append(compute_mean_valency(subsets, verbose=False))
        
    variance = np.var(mean_valency_values)
    print(variance)
    return variance


def compute_mean_valency(subsets, verbose=False):
    """
    Parameters
    ----------
        subsets (list of sets)
    
    Return
    ------
        mean_valency (float)
    """
    
    ### Find the list of intersections
    # For every subset, find the subsets it has a non-zero intersection with
    print(subsets)
    list_of_intersections = find_intersections_lists(subsets)
    print(list_of_intersections)
    
    ### Find the valencies for each subset (i.e., the number of subsets it intersects with)
    valencies = [len(x) for x in list_of_intersections]
    mean_valency = np.mean(valencies)
    if verbose:
        print("valencies: ", valencies)
        print("mean_valency: ", mean_valency)
    
    return mean_valency



import random

def fix_small_sets(subsets, min_size=2, protected_positions=None):
    """
    Ensure that no subset in the list has fewer than `min_size` elements.
    Subsets at protected_positions will NOT be modified.

    Parameters:
    -----------
    subsets : list of set
        A list where each element is a set representing a subset of elements.
    min_size : int, optional
        The minimum allowed size for each subset (default is 2).
    protected_positions : list of int, optional
        Indices of subsets that must not be modified.

    This function modifies `subsets` in place.
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


                ####################
                ################
                ##################
                ##################
import copy
import random

def repair_subsets(subsets, U, mec_position):
    """
    mec_position (list of int)

    Try to fix subsets so that:
    - union covers U,
    - no duplicates across subsets,
    - no subset smaller than 2,
    - mean valency is met (not implemented here).

    Subsets in mec_position will NOT be modified.
    Returns fixed subsets or raises RuntimeError if fails.
    """

    new_subsets = copy.deepcopy(subsets)

    # Add missing elements
    print("... Add missing elements ...")
    current_union = set.union(*new_subsets)
    missing = U - current_union
    for elem in missing:
        # Choose a subset not in mec_position
        candidates = [s for i, s in enumerate(new_subsets) if i not in mec_position]
        random.choice(candidates).add(elem)

    print(f"    ec: {new_subsets}")

    # Remove duplicates
    print("... Removing duplicates ...")
    elem_to_subsets = {}
    for i, subset in enumerate(new_subsets):
        for elem in subset:
            elem_to_subsets.setdefault(elem, []).append(i)
    print("    elem_to_subsets", elem_to_subsets)

    for elem, subsets_that_have_elem in elem_to_subsets.items():
        # Choose the one to keep, preferring a fixed subset if possible
        keep = random.choice([i for i in subsets_that_have_elem if i in mec_position] or subsets_that_have_elem)
        for subset_idx in subsets_that_have_elem:
            if subset_idx != keep and subset_idx not in mec_position:
                new_subsets[subset_idx].discard(elem)

    print(f"    ec = {new_subsets}")

    # Fix small sets (only outside mec_position)
    print("... Fixing small sets ...")
    fix_small_sets(new_subsets, protected_positions=mec_position)
    print("    ec", new_subsets)

    return new_subsets

  
if __name__ == "__main__":
    import copy
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    from utils_to_study_an_instance import find_intersections_lists, build_instance_graph

    from Wang_instances import info_dim6 as info_dim6_wang
    fixed_variance = compute_mean_valency_variance(info_dim6_wang)
    
    # Fixed data.
    U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    n = 6
    mec_len = 2
    max_w = 10 # maximum weight of mec subsets
    min_w = 2 # minimum weight of mec subsets
    max_w_non_mec = 4 # keep them small
    min_w_non_mec = 2
    elems_to_be_added = max_w_non_mec - min_w_non_mec
    

    NUM_INSTANCES = 2 # number of instances to be generated
    instance_dict = {} 
    for instance_idx in range(NUM_INSTANCES):
        print(f"    Instance {instance_idx}")
        START_AGAIN = True

        # Generate the MEC with 2 subsets.
        subsets = []
        w = random.randint(min_w, max_w)
        set_1 = set(random.sample(sorted(U), w))
        set_2 = U - set_1
        subsets.extend([set_1, set_2])
        mec = [set_1, set_2]
        print("    mec:", mec)


        while START_AGAIN == True:
            print("\n        **********************************")
            print("        while START_AGAIN == True:")

            # Generate the other subsets.
            not_mec = []
            for _ in range(n - mec_len):
                # For now, each subset has only min_w_non_mec=2 elements.
                random_two_elem_set = set(random.sample(sorted(U), min_w_non_mec))
                not_mec.append(random_two_elem_set)
            print("        not mec:", not_mec)


            # Shuffle the 'not_mec' part of subsets,
            subsets = mec + random.sample(not_mec, len(not_mec))
            subsets_tmp = copy.deepcopy(subsets)
            print(f"        shuffled subsets: {subsets_tmp}\n")

            for ith_elem_to_be_added in range(elems_to_be_added):
                print("            **********************************")
                print(f"            Adding element number #{1+ith_elem_to_be_added}")
                
                for subset in subsets_tmp[2::]: # exlcuding the mec
                    print("\n                **********************************")
                    print("                ...Working on subset:", subset)

                    # Find which elements 1-12 are not in 'subset' yet.
                    not_yet = list(U - subset)
                    subset.add(random.choice(not_yet))
                    print("                new subset:", subset)
                    
                    # See how the global mean valency has changed.
                    mean_valency = compute_mean_valency(subsets_tmp)
                    print("                updated global mean_valency: ", mean_valency, end="")

                    if (2 < mean_valency < 5):
                        print(" -->> **** OK, WE CAN CONTINUE ****")
                        print("                new subsets:", subsets_tmp)
                        # continue
                    else:
                        # This value of ith_elem_to_be_added will be a signal in the outer loop
                        # that the computation has NOT ended successfully.
                        ith_elem_to_be_added = elems_to_be_added + 1
                        print(" -->> #### FAIL OF MEAN VALENCY ####\n...Breaking the subset loop...")
                        break
                
                if ith_elem_to_be_added == elems_to_be_added - 1:
                    # The computation has ended successfully.
                    START_AGAIN = False
                    print("\n\n            ****END: EVERY ELEMENT HAS BEEN ADDED****")
                    print("            ****Breaking the ith_elem_to_be_added loop" 
                          + "(START_AGAIN = False)...****")
                    break
                else:
                    print(f"\n\n            ****MORE ELEMENTS TO BE ADDED****")
                    
        print("        ----------FINAL VALUES:----------")
        print("        subsets_tmp:", subsets_tmp)
        print("        subsets:", subsets)
        print("        mean_valency: ", mean_valency)
        print("        -----------------------------------")
        
        subsets = subsets_tmp

        # Now we generate an EC by selecting 3 or 4 subsets (not both the 1st and the 2nd)
        while True:
            ec_length = random.randint(3, 4)  # how many subsets to select
            ec = random.sample(subsets, ec_length)
            
            # Sort as they appear in subsets
            ec = sorted(ec, key=lambda x: subsets.index(x))
            positions = [i for i,s in enumerate(subsets) if s in ec]
            
            # Constraint: do not select both the first two elements
            if not all(mec_subset in ec for mec_subset in mec):
                break

        print("Selected subsets to create ec:", ec)
        print("Positions:", positions)
        
        mec_position = [i for i,s in enumerate(ec) if s in mec]
        print("mec Position:", mec_position)
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            ec = repair_subsets(ec, U, mec_position)
            subsets_tmp = np.array(copy.deepcopy(subsets))
            for i,subset in zip(positions,ec):
                subsets_tmp[i] = subset
            
            # Condition on global mean valency
            mv = compute_mean_valency(subsets_tmp)
            print(f"Attempt {attempt}: mean_valency = {mv}\n")
          
            if 2 < mv < 5 and all(len(s) >= 2 for s in subsets_tmp):
                print("SUCCESS: breaking...")
                break
            else:
                # If no valid configuration found after all attempts
                raise RuntimeError(f"Could not repair subsets in {max_attempts} attempts")
            
        print("subsets_tmp =", subsets_tmp)
        
#         print("final EC", ec)
#         subsets = remaining_subsets + ec
#         print("final subsets", subsets)
#         # subsets = random.sample(subsets, len(subsets))
#         random.shuffle(subsets)

#         print("final shuffled subsets", subsets)
#         print("final mean valency:",compute_mean_valency(subsets))

#         ec_str = ''.join(['1' if s in fixed_ec else '0' for s in subsets])
#         mec_str = ''.join(['1' if s in mec else '0' for s in subsets])

#         print("ec_str:", ec_str)
#         print("mec_str:", mec_str)
    
    
#         instance_dict[instance_idx] = {
#             'exact_covers': [ec_str, mec_str],
#             'mec': mec_str,
#             'subsets': subsets
#         }
        
#     instance_dict['U'] = U
#     print(instance_dict)
