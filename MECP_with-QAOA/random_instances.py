info_dim6 = {1: {'exact_covers': ['101110', '010001'],
     'mec': '010001',
     'subsets': [{9, 6}, {9, 10, 6, 1}, {5, 12}, {2, 3, 4, 10}, {1, 7, 8, 11}, {2, 3, 4, 5, 7, 8, 11, 12}]},
 2: {'exact_covers': ['010011', '001100'],
     'mec': '001100',
     'subsets': [{12, 5, 6, 7}, {1, 5, 7, 8, 9, 12}, {1, 2, 3, 5, 8, 9, 10, 11, 12}, {4, 6, 7}, {4, 6}, {2, 3, 10, 11}]},
 3: {'exact_covers': ['110011', '001100'],
     'mec': '001100',
     'subsets': [{8, 3, 6}, {1, 4, 5, 12}, {1, 11}, {2, 3, 4, 5, 6, 7, 8, 9, 10, 12}, {11, 10, 7}, {2, 9}]},
 4: {'exact_covers': ['011010', '000101'],
     'mec': '000101',
     'subsets': [{2, 11, 4, 5}, {8, 10, 12}, {9, 4, 5, 6}, {1, 2, 3, 4, 5, 7, 8, 9, 10, 11}, {1, 2, 3, 7, 11}, {12, 6}]},
 'U': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}



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
    list_of_intersections = find_intersections_lists(subsets)
    
    ### Find the valencies for each subset (i.e., the number of subsets it intersects with)
    valencies = [len(x) for x in list_of_intersections]
    mean_valency = np.mean(valencies)
    if verbose:
        print("valencies: ", valencies)
        print("mean_valency: ", mean_valency)
    
    return mean_valency



def fix_small_sets(subsets, min_size=2):
    """
    Ensure that no subset in the list has fewer than `min_size` elements.
    If a subset is too small, randomly borrow elements from other subsets
    that have more than `min_size` elements until all subsets satisfy the minimum size.

    Parameters:
    -----------
    subsets : list of set
        A list where each element is a set representing a subset of elements.
    min_size : int, optional
        The minimum allowed size for each subset (default is 2).

    This function modifies `subsets` in place.
    """
    changed = True
    while changed:
        changed = False
        # Iterate over all subsets to check their sizes
        for i, s in enumerate(subsets):
            if len(s) < min_size:
                # Find donor subsets with more than min_size elements to borrow from
                donors = [j for j, other in enumerate(subsets) if j != i and len(other) > min_size]
                if not donors:
                    # No donors available, cannot fix this subset now
                    continue
                # Randomly select a donor subset
                donor = random.choice(donors)
                # Randomly pick an element from the donor subset
                elem = random.choice(list(subsets[donor]))
                # Remove the element from the donor and add it to the small subset
                subsets[donor].remove(elem)
                subsets[i].add(elem)
                # Mark that a change was made, so we need to check again
                changed = True


def repair_subsets(subsets, max_attempts=1000):
    """
    Try up to max_attempts to fix subsets so that:
    - union covers 1 to 12,
    - no duplicates across subsets,
    - no subset smaller than 2,
    - mean valency is met.

    Returns fixed subsets or raises RuntimeError if fails.
    """
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        new_subsets = copy.deepcopy(subsets)

        # Add missing elements
        current_union = set.union(*new_subsets)
        missing = U - current_union
        for num in missing:
            random.choice(new_subsets).add(num)

        # Remove duplicates
        element_to_subsets = {}
        for i, s in enumerate(new_subsets):
            for num in s:
                element_to_subsets.setdefault(num, []).append(i)

        for num, indices in element_to_subsets.items():
            keep = random.choice(indices)
            for i in indices:
                if i != keep:
                    new_subsets[i].discard(num)

        # Fix
        fix_small_sets(new_subsets)

        # Condition
        if 2 < compute_mean_valency(new_subsets) < 5 and all(len(s) >= 2 for s in new_subsets):
            return new_subsets

        return new_subsets

    raise RuntimeError(f"Could not repair subsets in {max_attempts} attempts")


if __name__ == "__main__":
    import copy
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    from utils_to_study_an_instance import find_intersections_lists, build_instance_graph


    U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    n = 6

    mec_length = 2
    max_len_of_mec_subsets = 10
    min_len_of_mec_subsets = 2

    max_len_of_non_mec_subsets = 4 # keep them small

    # 2 is the min num of elements in non mec subsets
    elements_to_be_added = max_len_of_non_mec_subsets - 2 
    iteration = 0
    START_AGAIN = True

    # Generate the MEC with 2 subsets.
    subsets = []
    set_1 = set(random.sample(sorted(U), random.randint(min_len_of_mec_subsets, max_len_of_mec_subsets)))
    set_2 = U - set_1
    subsets.extend([set_1, set_2])
    mec = [set_1, set_2]
    print("mec:", mec)


    while START_AGAIN == True:
        print("    **********************************")

        # Generate the other subsets, with only 2 element each.
        not_mec = []
        for _ in range(n - mec_length):
            random_two_elem_set = set(random.sample(sorted(U), 2))
            not_mec.append(random_two_elem_set)
        print("    not mec:", not_mec)

            
        # Shuffle the "not_mec" part of subsets,
        subsets = mec + random.sample(not_mec, len(not_mec))
        subsets_tmp = copy.deepcopy(subsets)
        print("    shuffled subsets: ", subsets)

        for iteration in range(elements_to_be_added):
            for subset in subsets_tmp[2::]:
                print("        **********************************")
        
                print("        Working on subset:", subset)

                # Find which elements 1-12 are not in subset yet.
                available = U - subset
                elem = random.choice(list(available))
                subset.add(elem)
                print("        new subset:", subset)

                mean_valency = compute_mean_valency(subsets_tmp)
                print("        mean_valency: ", mean_valency)
                
                if (2 < mean_valency < 5):
                    subsets = subsets_tmp
                    print("\n        **** SUCCESS ****")
                    print("        new subsets:", subsets)
                    continue
                else:
                    # This value of iteration will be a signal in the outer loop
                    # that the computation has NOT ended successfully.
                    iteration = elements_to_be_added + 1
                    print("\n        **** FAIL ****")
                    break
        
        if iteration == elements_to_be_added - 1:
            # The computation has ended successfully.
            START_AGAIN = False

        print("----------\nFINAL VALUES:")
        print("subsets:", subsets)
        print("mean_valency: ", mean_valency)


    # Now we generate an EC by selecting 3 or 4 subsets (not both the 1st and the 2nd)
    while True:
        ec_length = random.randint(3, 4)  # how many subsets to select
        ec_subsets = random.sample(subsets, ec_length)
        

        # Constraint: do not select both the first two elements
        if not any(subset in ec_subsets for subset in mec):
            break

    print("Selected subsets to create ec:", ec_subsets)
    remaining_subsets = [s for s in subsets if not(s in ec_subsets)]

    fixed_ec = repair_subsets(ec_subsets)
    print("final EC", fixed_ec)
    subsets = remaining_subsets + fixed_ec
    print("final subsets", subsets)
    # subsets = random.sample(subsets, len(subsets))
    random.shuffle(subsets)
    
    print("final shuffled subsets", subsets)
    print("final mean valency:",compute_mean_valency(subsets))

    ec_str = ''.join(['1' if s in fixed_ec else '0' for s in subsets])
    mec_str = ''.join(['1' if s in mec else '0' for s in subsets])

    print("ec_str:", ec_str)
    print("mec_str:", mec_str)