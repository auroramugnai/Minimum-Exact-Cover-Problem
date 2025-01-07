from __future__ import annotations

import itertools 
import matplotlib.pyplot as plt 
import rustworkx as rx

# import networkx as nx  

# from utils_for_plotting_and_reading import highlight_correct_ticks

# def highlight_correct_ticks(ax, EXACT_COVERS: list) -> None:
#     """
#     Highlights ticks on the x-axis of a given matplotlib axis (`ax`) based on the states provided.

#     - If the state corresponds to the `EXACT_COVERS`, the tick is highlighted in crimson.
#     - If the state is the Minimum Exact Cover (MEC), the tick is highlighted in lime green.
#     - If the state is a one-hot state (1 surrounded by 0s), the tick is highlighted in light grey.
#     - Other ticks remain black.

#     Parameters:
#     -----------
#     ax : matplotlib.axes.Axes
#         The matplotlib axis containing the x-axis ticks to be styled.
#     EXACT_COVERS : list
#         A list of states representing exact covers to highlight on the axis.

#     Returns:
#     --------
#     None
#         This function modifies the `ax` object in place and does not return any value.
#     """
#     # Extract the current tick labels from the x-axis as text
#     xlabels = [elem.get_text() for elem in ax.xaxis.get_ticklabels()]

#     # Length of MEC used to generate one-hot states
#     n = len(MEC)

#     # Generate all distinct one-hot state permutations
#     one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]

#     # Iterate over the tick labels and apply the appropriate color
#     for (state, ticklbl) in zip(xlabels, ax.xaxis.get_ticklabels()):
#         ticklbl.set_color(
#             'limegreen' if state == MEC  # Highlight MEC in lime green
#             else 'crimson' if state in EXACT_COVERS  # Highlight exact covers in crimson
#             else 'lightgrey' if state in one_one_states  # Highlight one-hot states in light grey
#             else 'black'  # Default color for other states
#         )



#############################################################################################################
#############################################################################################################
#############################################################################################################

def define_instance(n, instance, verbose):
    """
    Parameters
    ----------
        n (int): instance dimension.
        instance (int): number of the instance.
        verbose (bool): if True, print is activated.

    Return
    ------
        U (set): set Universe of the instance.
        subsets_dict (dict): dictionary that enumerates subsets of the instance.
    """
    all_instances = {}

    if n == 6:
        all_instances = {
        1: [{4, 6, 7, 9, 10, 11}, {1, 2, 5, 6, 11, 12}, {8, 1, 12}, 
            {2, 3, 5}, {1, 3, 4, 5, 9, 12}, {2, 6, 7, 9, 12}],
        2: [{2, 11, 12, 6}, {2, 4, 6, 8, 9, 11}, {1, 3, 5, 7, 10, 12}, 
            {2, 7}, {2, 3, 4, 5, 8, 12}, {1, 2, 8, 9, 12}],
        3: [{8, 10, 3}, {2, 4, 5, 6, 9, 11, 12}, {1, 3, 7, 8, 10}, 
            {3, 4, 6, 8, 11}, {2, 3, 4, 6, 7, 9, 12}, {1, 7}],
        4: [{1, 4, 5, 6, 8, 11, 12}, {3, 6, 8, 9, 10}, {1, 2, 11, 5}, 
            {2, 3, 7, 9, 10}, {8, 3, 12, 6}, {4, 7, 12}],
        5: [{11, 7}, {1, 2, 4, 5, 9, 11}, {1, 3, 4, 6, 8}, 
            {2, 5}, {3, 6, 7, 8, 10, 12}, {9, 10, 12}],
        6: [{2, 3, 4, 5, 7, 8, 9, 10, 11, 12}, {5, 9, 10, 11}, {3, 6, 7, 8}, 
            {1, 2, 4, 12}, {1, 6}, {2, 3, 4, 7, 8, 12}],
        7: [{1, 2, 5, 6, 7, 9, 12}, {10, 3, 4}, {2, 5, 7, 8, 9, 10, 11}, 
            {8, 11}, {1, 7, 8, 11, 12}, {11, 9, 3, 7}],
        8: [{1, 2, 4, 5, 6, 8, 9, 10}, {1, 4, 6, 7}, {5, 8, 9, 11, 12}, 
            {4, 7, 8, 9, 10, 11, 12}, 
            {2, 3, 4, 5, 6, 7, 11, 12}, {10, 2, 3}],
        9: [{1, 4, 7, 9, 10, 11, 12}, {2, 3, 5, 6, 8}, {1, 2, 4, 9, 12}, 
            {2, 3, 6, 7, 8, 9}, {1, 4, 6, 7, 8, 10, 11}, {8, 11, 3, 6}],
        10: [{1, 11, 12, 6}, {2, 5, 8, 10, 11, 12}, {2, 6, 7, 8, 9, 10, 11}, 
             {2, 3, 4, 5, 7, 8, 9, 10}, {1, 2, 4, 5, 6, 7, 8, 11}, {1, 5, 6, 9, 10, 11, 12}]
        }

    elif n == 8:
        all_instances = {
        1: [{1, 2, 3, 4, 7, 8, 9, 12, 13, 14}, {1, 4, 5, 8, 9, 12, 13, 14, 16}, 
            {1, 6, 9, 10, 11, 13, 15, 16}, {5, 6, 10, 11, 15, 16}, 
            {1, 6, 7, 8, 9, 10, 14, 15}, {8, 13, 4, 12}, 
            {2, 3, 4, 5, 7, 8, 11, 12, 13}, {4, 5, 6, 7, 10, 12, 14, 16}],
        2: [{1, 2, 3, 5, 7, 12, 15}, {6, 7, 8, 9}, 
            {4, 6, 8, 9, 10, 11, 13, 14, 16}, {3, 4, 5, 6, 7, 9, 10, 11, 12, 15},
            {4, 12, 14}, {2, 3, 7, 8, 10, 15}, 
            {1, 4, 6, 8, 9, 10, 12, 14, 15}, {1, 2, 3, 5, 10, 11, 13, 15, 16}],
        3: [{4, 7, 14, 15, 16}, {4, 5, 8, 9, 10, 13, 14, 16}, 
            {1, 3, 4, 6, 8, 9, 10, 12, 13, 14}, {1, 3, 5, 6, 10}, 
            {2, 4, 6, 9, 10, 11, 15}, {1, 2, 4, 5, 9, 13, 14, 15}, 
            {1, 2, 3, 6, 7, 11, 12, 15}, {2, 8, 9, 11, 12, 13}],
        4: [{2, 4, 5, 6, 9}, {13, 14, 15, 16}, 
            {1, 4, 5, 6}, {7, 8, 10, 12}, 
            {1, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16}, {2, 3, 9, 11}, 
            {3, 4, 5, 7, 8, 10, 12}, {3, 7, 8, 9, 10, 11, 14}],
        5: [{5, 8, 10, 11, 12, 13, 16}, {1, 2, 3, 4, 6, 7, 9, 14, 15}, 
            {1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16}, {3, 4, 9, 11, 13, 14, 16}, 
            {2, 3, 6, 10, 13, 16}, {1, 3, 8, 10, 11, 13, 14, 15}, 
            {1, 6, 8, 10, 11, 13, 14, 16}, {3, 4, 7, 10, 11, 13, 14, 15, 16}],
        6: [{1, 2, 5, 6, 7, 10, 12, 15}, {5, 6, 9, 10}, 
            {2, 7, 11, 12}, {3, 6, 7, 8, 9, 12, 15, 16}, 
            {1, 3, 4, 8, 13, 14, 15, 16}, {2, 3, 6, 7, 8, 10, 15, 16}, 
            {1, 2, 3, 4, 7, 8, 12, 13, 14}, {2, 3, 8, 11, 12, 14}],
        7: [{2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15}, {2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16}, 
            {10, 11, 15, 7}, {2, 3, 9, 10, 11, 14, 16}, 
            {1, 2, 7, 10, 11, 12, 13, 14, 15}, {1, 4, 9, 14, 16}, 
            {3, 4, 5, 9, 11, 12, 14, 16}, {2, 7, 9, 10, 11, 12, 13, 14, 15, 16}],
        8: [{1, 4, 8, 9, 10, 13, 14, 15, 16}, {1, 2, 4, 5, 6, 7, 9, 12, 13}, 
            {2, 3, 5, 6, 7, 11, 12}, {5, 11, 12}, 
            {1, 2, 3, 7, 8, 10, 11, 13}, {1, 2, 4, 7, 8, 13, 14, 15, 16}, 
            {2, 3, 6, 7}, {1, 3, 4, 6, 9, 11, 12, 14, 16}],
        9: [{2, 4, 14, 15}, {1, 3, 6, 8}, 
            {5, 7, 10, 13, 16}, {1, 2, 4, 6, 7, 10, 11, 13, 15, 16}, 
            {4, 6, 7, 11, 12, 13, 15}, {3, 5, 8, 9, 12, 14}, 
            {9, 11, 12}, {2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16}],
        10: [{2, 3, 5, 6, 7, 8, 9, 11, 13, 16}, {1, 4, 5, 7, 9, 12, 14}, 
             {6, 7, 11, 13, 14, 15, 16}, {6, 7, 8, 9, 11, 15, 16}, 
             {2, 5, 10, 12, 14}, {1, 5, 6, 7, 11, 12, 14}, 
             {2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16}, {1, 3, 4, 13}]
        }


    elif n == 10:
        all_instances = {
        1: [{1, 4, 7, 8, 9, 12, 13, 14, 15, 17}, {1, 3, 6, 8, 9, 13, 14, 16, 17}, 
            {2, 4, 5, 7, 10, 11}, {1, 4, 5, 8, 9, 11, 13, 14, 17, 19}, 
            {1, 5, 6, 10, 11, 13, 14, 16, 18, 19, 20}, {1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 16}, 
            {2, 3, 5, 6, 10, 11, 16, 18, 19, 20}, {12, 15, 18, 19, 20}, 
            {2, 4, 5, 6, 8, 9, 10, 11, 13, 15, 17, 19, 20}, {5, 7, 8, 10, 12, 13, 16, 17, 18, 19, 20}],
        2: [{1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 20}, {2, 3, 4, 5, 9, 10, 20}, 
            {1, 6, 7, 8, 11, 12, 13, 19}, {14, 15, 16, 17, 18}, 
            {3, 5, 6, 10, 16, 18, 20}, {4, 7, 8, 9, 12, 14, 17, 19}, 
            {3, 4, 6, 14, 16, 18, 19}, {1, 2, 11, 13, 15}, 
            {2, 6, 7, 10, 13, 14, 16, 17, 20}, {2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 20}],
        3: [{2, 3, 6, 7, 9}, {4, 5, 8, 12, 15}, 
            {10, 11, 13, 14, 17}, {1, 16, 18, 19, 20}, 
            {2, 10, 11, 13, 15, 16, 17, 18}, {2, 5, 18, 19, 20}, 
            {3, 4, 5, 6, 7, 8, 10, 11, 16, 17, 20}, {7, 8, 10, 11, 13, 14, 16, 17}, 
            {1, 3, 4, 6, 9, 12, 15}, {1, 2, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20}],
        4: [{1, 2, 5, 8, 10, 11, 13, 15, 18, 20}, {3, 4, 6, 7, 9, 12, 14, 16, 17, 19}, 
            {2, 5, 6, 11, 15, 17}, {1, 3, 4, 7, 8, 9, 10, 12}, 
            {13, 14, 16, 18, 19, 20}, {2, 5, 6, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20}, 
            {1, 2, 6, 8, 10, 11, 13, 14, 15, 16, 18, 19, 20}, {1, 2, 3, 4, 7, 10, 16, 17}, 
            {1, 2, 4, 5, 7, 8, 9, 11, 13, 14, 15, 17, 19}, {1, 2, 3, 4, 8, 9, 13, 15, 16, 17, 19}],
        5: [{2, 4, 5, 6, 9, 15, 16, 19}, {3, 7, 8, 12, 13, 14, 17}, 
            {1, 10, 11, 18, 20}, {2, 8, 9, 10, 12, 13, 14, 15, 16, 17}, 
            {1, 3, 4, 5, 6, 7, 11, 18, 19, 20}, {1, 3, 6, 7, 8, 9, 12, 14, 16, 19, 20}, 
            {1, 4, 6, 10, 11, 12, 18, 19}, {1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 20}, 
            {1, 2, 9, 11, 15}, {2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 19}],
        6: [{1, 2, 3, 4, 9, 10, 12, 13, 15, 19, 20}, {2, 3, 5, 6, 8, 10, 12, 17}, 
            {1, 4, 7, 9, 14, 15, 19}, {11, 13, 16, 18, 20}, 
            {2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17}, {1, 7, 10, 11, 12, 18, 19, 20}, 
            {1, 2, 5, 11, 13, 14, 19, 20}, {1, 5, 7, 9, 10, 11, 13}, 
            {12, 14, 18, 19, 20}, {2, 3, 4, 6, 8, 15, 16, 17}],
        7: [{2, 3, 6, 9, 10, 12, 14, 17, 20}, {1, 4, 10, 11, 12, 14, 15, 17, 18, 20}, 
            {2, 4, 5, 7, 8, 11, 12, 15, 17}, {2, 3, 4, 5, 6, 8, 9, 11, 14, 17, 18, 20}, 
            {1, 2, 4, 16, 17, 18, 20}, {1, 4, 5, 7, 8, 11, 13, 15, 16, 18, 19}, 
            {5, 6, 8, 10, 11, 13, 15}, {3, 7, 9, 12, 14, 19}, 
            {1, 2, 4, 6, 7, 8, 9, 12, 13, 15, 17, 18, 20}, {1, 5, 10, 12, 15, 17, 18, 19, 20}],
        8: [{4, 8, 12, 15, 16, 17, 18}, {1, 2, 4, 7, 8, 10, 11, 12, 13}, 
            {3, 5, 6, 12, 14, 18, 19, 20}, {6, 7, 9, 10, 13, 14, 19}, 
            {1, 2, 3, 5, 11, 20}, {1, 2, 4, 7, 8, 9, 10, 11, 13, 15, 16, 17}, 
            {1, 3, 6, 8, 10, 15, 17, 19}, {2, 3, 4, 10, 12, 13, 14, 16, 17, 20}, 
            {2, 3, 4, 8, 11, 12, 13, 18, 20}, {1, 6, 7, 9, 10, 11, 13, 14, 16}],
        9: [{2, 4, 5, 6, 7, 11, 15, 17}, {5, 9, 10, 14, 17, 18, 20}, 
            {4, 5, 6, 8, 9, 11, 13, 14, 16, 17, 20}, {4, 7, 8, 11, 13, 15, 19}, 
            {2, 5, 6, 7, 12, 14, 15, 16, 17}, {1, 2, 3, 6, 12, 16}, 
            {1, 2, 5, 6, 8, 10, 14, 16, 18, 19, 20}, {1, 3, 4, 8, 9, 10, 11, 13, 18, 19, 20}, 
            {1, 4, 5, 8, 9, 10, 12, 16, 17, 18, 20}, {1, 2, 3, 6, 7, 8, 9, 11, 14, 15, 16}],
        10: [{2, 3, 5, 7, 8, 9, 10, 12, 13, 14, 15}, {1, 2, 4, 7, 8, 10, 11, 14, 18}, 
             {1, 4, 5, 8, 9, 12, 13, 15, 16}, {1, 2, 3, 5, 9, 10, 14, 19}, 
             {1, 6, 7, 11, 12, 16, 18}, {3, 4, 6, 8, 11, 13, 14, 15, 19, 20}, 
             {1, 2, 5, 7, 10, 12, 13, 16, 17, 20}, {1, 3, 5, 6, 9, 12, 15, 16, 18, 19}, 
             {3, 7, 9, 10, 11, 12, 14, 15, 19}, {2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15}]
        }


    else:
        raise ValueError("Unsupported dimension `n`. Only dimensions 6, 8, and 10 are supported.")

    if instance not in all_instances:
        raise ValueError(f"Instance {instance} is not defined for dimension {n}.")

    subsets = all_instances[instance]
    U = set().union(*subsets)
    subsets_dict = {i + 1: subset for i, subset in enumerate(subsets)}

    if verbose:
        print(f"Universe: {U}")
        print("Subsets:")
        for idx, subset in subsets_dict.items():
            print(f"  {idx}: {subset}")

    return U, subsets_dict



#############################################################################################################
#############################################################################################################
#############################################################################################################

def build_instance_graph(subsets, verbose=False, draw_graph=False):
    """
    Builds a graph representation of the instance, where each subset is a node, and edges represent 
    non-zero intersections between subsets. Optionally, it can draw the graph and print additional details.

    Parameters
    ----------
        subsets (list of sets): List of sets defining the instance.
            Each set represents a subset of the instance.
        verbose (bool): If True, prints additional information such as subset lengths and valencies.
        draw_graph (bool): If True, draws the graph using matplotlib.

    Return
    ------
        list_of_intersections (list of lists): A list containing lists of intersections for each subset.
            Each list contains the indices of subsets that have a non-zero intersection with the current subset.

    """
    n = len(subsets)

    ### Build the graph of the instance
    # Generate edges based on the intersections between the sets in the subsets list
    edges = edges_from_sets(subsets)
    
    # Create a graph object
    G = rx.PyGraph()
    
    # Add nodes to the graph (one for each subset)
    G.add_nodes_from([i for i in range(1, n+1)])
    
    # Extend the graph with edges based on intersections
    G.extend_from_edge_list(edges)
    
    # G.remove_node(0)  # (Optional) This line seems to be commented out, possibly removing the first node
    
    # Draw the graph if requested
    if draw_graph:
        mpl_draw(
            G, pos=rx.circular_layout(G), with_labels=True, node_color="#EE5396", font_color="#F4F4F4"
        )

    ### Find the list of intersections
    # For every subset, find the subsets it has a non-zero intersection with
    list_of_intersections = find_intersections_lists(subsets)

    # Optional: Print intersections if needed (currently commented out)
    # for i,l in enumerate(list_of_intersections):
    #     print(f"Set #{i} intersections:", l)
    # print("list_of_intersections", list_of_intersections)

    if verbose:
        ### Find the lengths of each subset (i.e., the number of elements in each subset)
        lengths = [len(s) for s in subsets]
        print("lengths: ", lengths)

        ### Find the valencies for each subset (i.e., the number of subsets it intersects with)
        valencies = [len(x) for x in list_of_intersections]
        print("valencies: ", valencies)

    return list_of_intersections



#############################################################################################################
#############################################################################################################
#############################################################################################################

def find_intersections_lists(subsets):
    """
    Finds the intersections between the subsets in a list. For each subset, returns a list of indices of the subsets 
    that have a non-zero intersection with it.

    Parameters
    ----------
    subsets (list of sets): A list of sets for which we want to find intersections.

    Return
    ------
    list_of_intersections (list of lists): A list where the i-th element is a list containing the indices of 
    all the sets that have a non-zero intersection with the i-th set in `subsets`.

    Example
    -------
    "Set #0 intersections: [1, 5]" means that the 0th set has a non-zero intersection with the 1st and 5th set 
    of `subsets`.

    """
    list_of_intersections = []
    
    # Loop through each subset to find intersections
    for i in range(len(subsets)):
        ith_set_intersections = []
        
        # Check intersection with every other subset
        for j in range(len(subsets)):
            if i != j and subsets[i].intersection(subsets[j]):
                ith_set_intersections.append(j)
        
        # Append the list of intersections for the i-th subset
        list_of_intersections.append(ith_set_intersections)
    
    return list_of_intersections



#############################################################################################################
#############################################################################################################
#############################################################################################################

def edges_from_sets(subsets):
    """
    Creates a list of edges for a graph, where each edge connects two sets that have a non-zero intersection.

    Parameters
    ----------
    subsets (list of sets): A list of sets.

    Returns
    -------
    egs (list of tuples): A list of tuples representing the edges of a graph.
    Each tuple (i, j) indicates that subsets i and j have a non-zero intersection.

    Example
    -------
    (1, 3) means that the 1st and 3rd subsets have a non-zero intersection.

    """
    egs = []
    l = len(subsets)
    
    # Loop through all pairs of subsets
    for i in range(l):
        for j in range(i, l):
            # Find the intersection between subsets i and j
            w = subsets[i].intersection(subsets[j])
            
            # If intersection is non-zero and subsets are different, add edge
            if len(w) != 0 and i != j:
                egs.append((i, j))

    return egs



#############################################################################################################
#############################################################################################################
#############################################################################################################

def find_U_from_subsets(subsets_dict):
    """
    Finds the union of all sets in a dictionary of subsets.

    Parameters
    ----------
    subsets_dict (dict): A dictionary where keys are identifiers, and values are sets.

    Returns
    -------
    U (set): The union of all the sets in the dictionary.

    Example
    -------
    If `subsets_dict = {0: {1, 2}, 1: {2, 3}}`, the result would be `{1, 2, 3}`.

    """
    # Initialize U as the first set in the dictionary
    U = subsets_dict[0]
    
    # Union all sets in the dictionary
    for s in subsets_dict.values():
        U = U | s
    
    return U



#############################################################################################################
#############################################################################################################
#############################################################################################################

def bit_gen(n):
    """
    Generates all possible n-tuples of bits (binary tuples).

    Arguments
    ---------
    n : int
        The length of the tuples to be produced. Each tuple will consist of `n` bits, where each bit is either 0 or 1.

    Return
    ------
    generator
        A generator that yields all possible n-tuples of bits (from (0, 0, ..., 0) to (1, 1, ..., 1)).

    Example
    -------
    If n = 2, the generator will produce the following tuples:
    (0, 0), (0, 1), (1, 0), (1, 1).
    """
    return itertools.product([0, 1], repeat=n)



#############################################################################################################
#############################################################################################################
#############################################################################################################

def compute_energy_Lucas(x, U, subsets_dict):
    """
    Computes the energy of a binary array `x` based on the 
    Andrew Lucas formulation in "Ising formulations of many NP problems".

    Arguments:
    ----------
    x : str
        A binary array represented as a string (e.g., "110000").
    U : set
        A set representing the universe or the collection of elements.
    subsets_dict : dict
        A dictionary where keys represent the subset identifiers and the values are sets. The union of all sets
        in `subsets_dict` is equal to `U`.

    Returns:
    -------
    E : float
        The computed energy of the binary array `x` based on the Ising formulation described by Andrew Lucas.

    Example:
    --------
    If x = "110000", subsets_dict contains subsets of U, and U is the union of these subsets, 
    the function will return the computed energy of x.
    """
    # Convert the binary string `x` into a list of integers: "110000" -> [1, 1, 0, 0, 0, 0]
    x = [int(digit) for digit in x]

    A = 1.  # Constant A (could be used for scaling)

    E_A = 0.

    # For each element in the set U, compute the energy contribution
    for uu in U:
        # Sum the values in `x` where the indices belong to subsets that include `uu`
        counts = sum([x[i-1] for i in subsets_dict.keys() if uu in subsets_dict[i]])
        
        # Add to the energy based on the formula (1 - counts)^2
        E_A += (1 - counts)**2

    # Scale energy by the constant A
    E_A = A * E_A

    return E_A



#############################################################################################################
#############################################################################################################
#############################################################################################################

def compute_energy_Wang(state, U, subsets_dict, k=1):
    """
    Computes the energy of a binary array `state` as described by 
    Sha-Sha Wang et al. in "Quantum Alternating Operator Ansatz 
    for solving the Minimum Exact Cover Problem".

    Arguments:
    ----------
    state : str
        A binary array represented as a string (e.g., "110000").
    U : set
        A set representing the universe or collection of elements.
    subsets_dict : dict
        A dictionary where keys represent the subset identifiers, and values are sets. The union of all sets
        in `subsets_dict` is equal to `U`.
    k : int, optional
        A multiplicative factor used to compute the energy. It must be greater than 0. Default is 1.

    Returns:
    -------
    E : float
        The computed energy of the binary array `state` based on the formulation by Wang et al.

    Example:
    --------
    If state = "110000", subsets_dict contains subsets of U, and U is the union of these subsets,
    the function will return the computed energy of the state.
    """
    # Convert the binary string `state` into a list of integers: "110000" -> [1, 1, 0, 0, 0, 0]
    state = [int(digit) for digit in state]
    
    # Extract the list of subsets from the dictionary
    subsets = list(subsets_dict.values())

    n = len(state)  # The length of the state
    # Compute l2 and l1 as described in the formula
    l2 = 1 / (n * len(U) - 2)
    l1 = k * n * l2

    E = 0.
    # Loop through each position and digit in the state
    for pos, digit in enumerate(state):
        w = len(subsets[pos])  # The weight of the current subset
        # Update the energy according to the given formula
        E += l1 * w * digit - l2 * digit

    # Invert the energy as required
    E = -E

    return E



#############################################################################################################
#############################################################################################################
#############################################################################################################
def show_spectrum(n, instance, k, fontsize=13, verbose=False):
    """
    Shows the spectrum of an instance for a given value of k, including the energy of all states 
    and the energy of feasible states. Highlights the exact covers and minimal exact covers (MEC).

    Parameters
    ----------
    n : int
        The dimension of the instance.
    instance : int
        The index or identifier of the instance to consider.
    k : float
        The value of k chosen for the problem instance.
    verbose : bool, optional, default=False
        If True, additional print statements will be shown during the execution.
    fontsize : int, optional, default=13
        The fontsize to use in the figure.
    Example
    -------
    show_spectrum(10, 2, 1.0, verbose=True)
    """
    
    # Define the universe and subsets based on the given instance
    U, subsets_dict = define_instance(n, instance, verbose=verbose)

    # Find the spectrum, including all states, energies, feasible states, and exact covers
    states, energies, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, k)
    
    # Extract the Minimal Exact Covers (MEC), which are the exact covers with the minimum number of 1s
    MEC = [state for state in EXACT_COVERS if state.count("1") == min([x.count("1") for x in EXACT_COVERS])]

    # Print the exact covers and MEC if verbose mode is activated
    if verbose:
        print("EXACT_COVERS:", EXACT_COVERS)
        print("MEC:", MEC)
        
    
    from utils_for_plotting_and_reading import highlight_correct_ticks

    #############################################################################
    #### PLOT ALL STATES ENERGY
    plt.figure(figsize=(9,3))  
    plt.rcParams['font.size'] = fontsize 
    plt.title("All states")  
    plt.plot(states, energies, 
             marker='o', color='k', linestyle='dotted')  # Plot the states against their energies
    plt.xticks(rotation='vertical', fontsize=fontsize-2) 
    plt.xlabel("States")  
    plt.ylabel("Energy")  
    plt.grid() 
    
    # Highlight the exact covers on the plot
    highlight_correct_ticks(plt.gca(), EXACT_COVERS)

    plt.show()

    #############################################################################
    #### PLOT THE FEASIBLE STATES ENERGIES
    plt.figure(figsize=(7,3)) 
    plt.rcParams['font.size'] = fontsize  
    plt.title("Feasible states")
    plt.plot(states_feasible, energies_feasible, 
             marker='o', color='k', linestyle='dashed') 
    plt.xticks(rotation='vertical', fontsize=fontsize)
    plt.xlabel("States")
    plt.ylabel("Energy")
    plt.grid()  

    # Highlight the exact covers on the plot
    highlight_correct_ticks(plt.gca(), EXACT_COVERS)

    plt.show()




#############################################################################################################
#############################################################################################################
#############################################################################################################

def find_spectrum(U, subsets_dict, n, k):
    """
    Computes the spectrum of an instance, which includes:
    - All possible states and their associated energies.
    - Feasible states and their energies.
    - Exact covers that satisfy specific constraints.

    Parameters
    ----------
    U : set
        The universe set of the instance, containing all elements.
    subsets_dict : dict
        A dictionary where keys represent subset identifiers, and values are the corresponding sets.
    n : int
        The dimension of the instance, which determines the number of bits in each state.
    k : float
        The chosen parameter `k`, used in the energy computation.

    Returns
    -------
    states : list of str
        A list of all possible states represented as binary strings.
    energies : list of float
        A list of energies associated with each state.
    states_feasible : list of str
        A list of all feasible states, represented as binary strings.
    energies_feasible : list of float
        A list of energies associated with the feasible states.
    EXACT_COVERS : list of str
        A list of states that are exact covers, which satisfy certain constraints.

    Example
    -------
    If U = {1, 2, 3}, subsets_dict = {1: {1, 2}, 2: {2, 3}}, n = 3, and k = 1.0,
    the function will return a list of states, their energies, the feasible states,
    and the exact covers.
    """
    # Convert subsets_dict values to a list for easier iteration
    subsets = list(subsets_dict.values())

    states = []
    energies = []
    states_feasible = []
    energies_feasible = []
    EXACT_COVERS = []

    # Generate all possible n-bit tuples (binary strings)
    for nuple in bit_gen(n):
        # Convert the tuple to a binary string (e.g., (1, 0, 1) -> "101")
        state = "".join([str(bit) for bit in nuple])

        # Choose subsets where the corresponding bit in the state is 1
        chosen_subsets = [subsets[i] for i, digit in enumerate(state) if digit == "1"]
        u = set().union(*chosen_subsets)  # Union of the chosen subsets
        sum_of_len = sum([len(sub) for sub in chosen_subsets])  # Total length of chosen subsets

        # Check if the state satisfies the condition (i.e., subsets do not intersect)
        # This ensures the union of the selected subsets has the same size as the sum of their lengths.
        energy = compute_energy_Wang(state, U, subsets_dict, k)

        # If the state is feasible (i.e., no intersection between selected subsets), add it to the feasible lists
        if len(u) == sum_of_len:
            states_feasible.append(state)
            energies_feasible.append(energy)

        # Add the state and its energy to the respective lists
        states.append(state)
        energies.append(energy)

        #### CHECK IF STATE IS AN EXACT COVER
        E = compute_energy_Lucas(state, U, subsets_dict)

        # If the energy computed by Lucas is 0, it's an exact cover
        if E == 0:
            EXACT_COVERS.append(state)
    return states, energies, states_feasible, energies_feasible, EXACT_COVERS

