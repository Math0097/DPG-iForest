import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math
import os
import numpy as np

import graphviz
import networkx as nx

import hashlib
from joblib import Parallel, delayed

from collections import defaultdict
from functools import partial

from decimal import Decimal, ROUND_HALF_UP

#EDITED

def digraph_to_nx(graphviz_graph):
    '''
    This function converts a Graphviz directed graph (DiGraph) to a NetworkX directed graph (DiGraph).
    It also extracts node descriptions, edges with weights, and edges with labels (names of nodes and their frequency).

    Args:
    graphviz_graph: The input Graphviz directed graph.

    Returns:
    networkx_graph: The converted NetworkX directed graph.
    nodes_list: A sorted list of nodes with their descriptions.
    edges_label: A list of edges with node names and frequencies.
    '''

    # Create an empty directed graph in NetworkX
    networkx_graph = nx.DiGraph()

    # Initialize lists to store nodes, edges with weights, and edges with labels
    nodes_list = []
    edges = []
    weights = {}

    edges_label = []

    # Dizionario per mappare gli ID dei nodi ai loro label
    node_labels = {}
    
    # Inizializza una variabile per la somma delle frequenze
    total_freq = 0
    
    # Prima passata: Calcola la somma di tutte le frequenze degli archi
    for edge in graphviz_graph.body:
        # Cerca l'arco con la frequenza
        match_edge = re.match(r'\s*([0-9]+)\s*->\s*([0-9]+)\s*\[label=([0-9]+)', edge)
        if match_edge:
            freq = int(match_edge.group(3))  # Frequenza dell'arco
            total_freq += freq  # Aggiungi la frequenza alla somma totale

    # Controllo se abbiamo trovato qualche arco con frequenze
    if total_freq == 0:
        raise ValueError("Nessuna frequenza trovata negli archi.")

    # Extract nodes and edges from the Graphviz graph
    for edge in graphviz_graph.body:
        # Check if the line represents a node (contains '[label=')
        match_node = re.match(r'\s*([0-9]+)\s*\[label="([^"]+)"', edge)
        if match_node:
            node_id = match_node.group(1).strip()
            node_label = match_node.group(2).strip()
            node_labels[node_id] = node_label  # Mappa l'ID del nodo al suo label
            nodes_list.append([node_id, node_label])  # Aggiungi nodo alla lista dei nodi

        # Check if the line represents an edge (contains '->')
        if "->" in edge:
            # Extract source and destination nodes
            src, dest = edge.split("->")
            src = src.strip()
            dest = dest.split(" [label=")[0].strip()

            # Initialize weight to None
            weight = None

            # Extract weight from edge attributes if available
            if "[label=" in edge:
                attr = edge.split("[label=")[1].split("]")[0].split(" ")[0]
                weight = (
                    float(attr)
                    if attr.isdigit() or attr.replace(".", "").isdigit()
                    else None
                )
                weights[(src, dest)] = weight  # Store weight for the edge

            # Add the edge to the list
            edges.append((src, dest))

            # Cerca l'arco con la frequenza
            match_edge = re.match(r'\s*([0-9]+)\s*->\s*([0-9]+)\s*\[label=([0-9]+)', edge)
            if match_edge:
                nodo_da = match_edge.group(1)  # ID del nodo di partenza
                nodo_a = match_edge.group(2)   # ID del nodo di arrivo
                freq = int(match_edge.group(3))  # Frequenza dell'arco
                
                # Calcola la percentuale della frequenza
                freq_percent = (freq / total_freq) * 100

                # Mappare gli ID dei nodi ai rispettivi label
                src_label = node_labels.get(nodo_da, nodo_da)
                dest_label = node_labels.get(nodo_a, nodo_a)

                # Aggiungere l'arco con label e frequenza alla lista edges_label
                edges_label.append((src_label, dest_label, round(freq, 2)))

    # Sort edges and nodes
    edges = sorted(edges)
    nodes_list = sorted(nodes_list, key=lambda x: x[0])

    # Add nodes and edges to the NetworkX graph
    for edge in edges:
        src, dest = edge
        # Add edge with weight if available, else add without weight
        if (src, dest) in weights:
            networkx_graph.add_edge(src, dest, weight=weights[(src, dest)])
        else:
            networkx_graph.add_edge(src, dest)

    # Return the constructed NetworkX graph, the list of nodes, and the labeled edges
    return networkx_graph, nodes_list, edges_label

def tracing_if(case_id, sample, iforest, feature_names, decimal_threshold, mode_graph, max_depth, mode):
    """
    Traccia i percorsi decisionali per ogni iTree presente in un Isolation Forest per un determinato campione.
    Registra il percorso decisionale (confronti effettuati ad ogni nodo) e la classe risultante (inlier o outlier).

    Args:
        case_id: Identificativo del campione.
        sample: Il campione di input.
        iforest: L'Isolation Forest contenente gli iTrees.
        feature_names: I nomi delle features usate negli alberi.
        decimal_threshold: Numero di decimali a cui arrotondare le soglie.
        mode_graph: Modalità di output del grafico ("last" per mostrare solo l'ultimo nodo, ecc.).
        max_depth: Profondità massima considerata.
        mode: Modalità ("global" o altro) che influenza la classificazione in foglia.

    Returns:
        event_log: Una lista degli step decisionali per ogni albero.
    """
    event_log = []
    sample = sample.reshape(1, -1)
    
    # Calcola il punteggio del campione una sola volta
    score = iforest.decision_function(sample)
    
    def build_path(tree, node_index, path, depth):
        node = tree.tree_
        is_leaf = node.children_left[node_index] == node.children_right[node_index]

        # Gestione della classificazione in foglia in base alla modalità
        if mode == "global":
            if is_leaf:
                if score < 0 and depth < max_depth:
                    path.append("Class -1")
                else:
                    path.append("Class 1")
                return
        else:
            if is_leaf:
                if score < 0 and depth < max_depth:
                    path.append("Class -1")
                else:
                    path.append("Class 1")
                return

        # Elaborazione del nodo interno
        feature_index = node.feature[node_index]
        feature_name = feature_names[feature_index]
        threshold = round(node.threshold[node_index], decimal_threshold)
        sample_val = sample[0, feature_index]
        go_left = sample_val <= threshold
        next_node = node.children_left[node_index] if go_left else node.children_right[node_index]

        # Determina il peso e la condizione da registrare
        freq_pes = 0 if score < 0 else 1
        condition = (f"{feature_name} <= {threshold} {freq_pes} {depth}"
                     if go_left else
                     f"{feature_name} > {threshold} {freq_pes} {depth}")
        path.append(condition)

        # Continua ricorsivamente nel prossimo nodo
        build_path(tree, next_node, path, depth + 1)

    # Cicla attraverso gli alberi dell'Isolation Forest
    for i, tree in enumerate(iforest.estimators_):
        sample_path = []
        build_path(tree, 0, sample_path, 0)

        # Registra gli eventi per l'albero corrente
        tree_events = [[f"sample{case_id}_dt{i}", step] for step in sample_path]
        event_log.extend(tree_events)

    return event_log


def filter_log(log, perc_var=0.0000000000000000000000000000001, n_jobs=-1):
    
    """
    Filters a log based on the variant percentage. Variants (unique sequences of activities for cases) 
    that occur less than the specified threshold are removed from the log.

    Args:
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.
    perc_var: A float representing the minimum percentage of total traces a variant must have to be kept.
    n_jobs: Number of parallel jobs to use. Default is -1 (use all available CPUs).

    Returns:
    log: A filtered pandas DataFrame containing only the cases and activities that meet the variant percentage threshold.
    """

    def process_chunk(chunk):
        # Filter the log DataFrame to only include cases from the current chunk
        filtered_log = log[log['case:concept:name'].isin(chunk) & log['concept:name'].apply(lambda x: not isinstance(x, (int, float)))]
        grouped = filtered_log.groupby('case:concept:name')['concept:name'].agg('|'.join)
        # Invert the series to a dictionary where keys are the concatenated 'concept:name' and values are lists of cases
        chunk_variants = {}
        for case, key in grouped.items():
            if key in chunk_variants:
                chunk_variants[key].append(case)
            else:
                chunk_variants[key] = [case]
        
        return chunk_variants

    # Split the cases into chunks for parallel processing
    cases = log["case:concept:name"].unique()
    
    # If n_jobs is -1, use all available CPUs, otherwise use the provided n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count()  # Get the number of available CPU cores
    
    # Adjust n_jobs if there are fewer cases than n_jobs
    n_jobs = min(n_jobs, len(cases))  # Ensure n_jobs is not larger than the number of cases

    # Calculate chunk size
    chunk_size = len(cases) // n_jobs if len(cases) // n_jobs > 0 else 1  # Ensure chunk_size is at least 1
    
    # Split the cases into chunks
    chunks = [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]
    
    # Process each chunk in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)

    # Combine results into a single dictionary
    variants = {}
    for result in results:
        for key, value in result.items():
            if key in variants:
                variants[key].extend(value)
            else:
                variants[key] = value

    # Get the total number of unique traces in the log
    total_traces = log["case:concept:name"].nunique()

    # Helper function to filter variants in parallel
    def filter_variants(chunk):
        local_cases, local_activities = [], []
        for k, v in chunk.items():
            if len(v) / total_traces >= perc_var:
                for case in v:
                    for act in k.split("|"):
                        local_cases.append(case)
                        local_activities.append(act)
        return local_cases, local_activities

    # Split the dictionary of variants into chunks for filtering
    variant_items = list(variants.items())
    
    # Split variant_items into chunks
    chunk_size = len(variant_items) // n_jobs if len(variant_items) // n_jobs > 0 else 1  # Ensure chunk_size is at least 1
    chunks = [variant_items[i:i + chunk_size] for i in range(0, len(variant_items), chunk_size)]
    
    # Process filtering in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(filter_variants)(dict(chunk)) for chunk in chunks)

    # Combine results into lists of cases and activities
    cases, activities = [], []
    for local_cases, local_activities in results:
        cases.extend(local_cases)
        activities.extend(local_activities)

    # Ensure both lists are of the same length before creating DataFrame
    assert len(cases) == len(activities), f"Length mismatch: {len(cases)} cases vs {len(activities)} activities"

    # Create a new DataFrame from the filtered cases and activities
    filtered_log = pd.DataFrame(zip(cases, activities), columns=["case:concept:name", "concept:name"])

    return filtered_log





def discover_dfg(log, predicates, mode_score, max_depth, n_inliers, n_outliers, n_jobs=-1):

    def feature(condition):

        # Regex per trovare il nome della feature, l'operatore < o > e il valore
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)\s*(<=|<|>|>=)\s*(-?[\d.]+)\s*(-?[\d.]+)\s*(-?[\d.]+)", condition)
        
        if match:
            feature_name = match.group(1)
            value = float(match.group(3))
            score = float(match.group(4))
            depth = int(match.group(5))    
            return f"{feature_name} ", value, score, depth    # Lo spazio è FONDAMENTALE
        else:
            return condition, None, None, None
    
    def feature_operator(condition):
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)\s*(<=|<|>|>=)\s*(-?[\d.]+)\s*(-?[\d.]+)\s*(-?[\d.]+)", condition)

        if match:
            feature_name = match.group(1)
            operator = match.group(2)
            value = float(match.group(3))
            score = float(match.group(4))
            depth = int(match.group(5))
            return f"{feature_name} {operator}", value, score, depth
        else:
            return condition, None, None, None
        
    def feature_operator_depth(condition):

        condition = condition.replace('<=', '<').replace('>=', '>')
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)\s*(<=|<|>|>=)\s*(-?[\d.]+)\s*(-?[\d.]+)\s*(-?[\d.]+)", condition)

        if match:
            feature_name = match.group(1)
            operator = match.group(2)
            value = float(match.group(3))
            score = float(match.group(4))
            depth = int(match.group(5))
            return f"{feature_name} {operator} {depth}", value, score, depth
        else:
            return condition, None, None, None

    function_dict = {
        'feature': feature,
        'feature_operator': feature_operator,
        'feature_operator_depth': feature_operator_depth,
    }

    log['concept:name'], log['concept:name:value'], log['concept:name:score'], log['concept:name:depth'] = zip(*log['concept:name'].apply(function_dict[predicates]))
    
    
    """
    Mines the nodes and edges relationships from an event log and returns a dictionary representing
    the Data Flow Graph (DFG). The DFG shows the frequency of transitions between activities.

    Args:
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.
    n_jobs: Number of parallel jobs to use. Default is -1 (use all available CPUs).

    Returns:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.
    """
    
    # Helper function to process a chunk of cases
    def process_chunk(chunk, mode_score):
            chunk_dfg = {}
            # Dizionario separato per il conteggio delle transizioni
            count_dfg = {}
            if mode_score == "freq_pes": #più è piccolo meglio è
                for case in chunk:
                    # Extract the trace (sequence of activities) for the current case
                    trace_df = log[log["case:concept:name"] == case].copy()
                    trace_df.sort_values(by="case:concept:name", inplace=True)
                    
                    score_value = trace_df["concept:name:score"].iloc[0]
                    if score_value == 0: # outliers
                        # multiplier = 0
                        multiplier = Decimal((n_outliers + n_inliers) / n_outliers).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                        # multiplier = Decimal(1 / n_outliers).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                        # multiplier = n_inliers
                    elif score_value == 1: # inliers
                        # multiplier = 1
                        multiplier = Decimal((n_outliers + n_inliers) / n_inliers).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                        # multiplier = Decimal(1 / n_inliers).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                        # multiplier = n_outliers
                    else:
                        continue

                    # Iterate through the trace to capture transitions between consecutive activities
                    for i in range(len(trace_df) - 1):
                        key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])  # Transition
                        
                        # score = Decimal(trace_df.iloc[i, 3]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                        
                        if key in chunk_dfg:
                            chunk_dfg[key] += multiplier  # Increment count if transition exists
                        else:
                            chunk_dfg[key] = multiplier  # Initialize count if transition is new
            
            elif mode_score == "freq": #più è grande meglio è
                for case in chunk:
                    # Extract the trace (sequence of activities) for the current case
                    trace_df = log[log["case:concept:name"] == case].copy()
                    trace_df.sort_values(by="case:concept:name", inplace=True)

                    # Iterate through the trace to capture transitions between consecutive activities
                    for i in range(len(trace_df) - 1):
                        key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])  # Transition
                        
                        if key in chunk_dfg:
                            chunk_dfg[key] += 1  # Increment count if transition exists
                        else:
                            chunk_dfg[key] = 1  # Initialize count if transition is new
                            
            elif mode_score == "depth": #più è piccolo meglio è
                for case in chunk:
                    # Extract the trace (sequence of activities) for the current case
                    trace_df = log[log["case:concept:name"] == case].copy()
                    trace_df.sort_values(by="case:concept:name", inplace=True)

                    # Iterate through the trace to capture transitions between consecutive activities
                    for i in range(len(trace_df) - 1):
                        key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])  # Transition
                        depth = trace_df.iloc[i, 4] # Use the last depth value for all transition
                        
                        # Calcola (max_depth - depth) per la transizione
                        depth_value = max_depth - depth
                        
                        if key in chunk_dfg:
                            # Se la transizione esiste già, aggiorna la media incrementale
                            current_avg = chunk_dfg[key]  # Estrai la media corrente
                            count = count_dfg[key]  # Estrai il conteggio corrente
                            new_avg = (current_avg * count + depth_value) / (count + 1)  # Calcola la nuova media
                            chunk_dfg[key] = round(new_avg, 2)  # Aggiorna solo con la nuova media
                            count_dfg[key] = count + 1  # Incrementa il conteggio
                        else:
                            # Inizializza con il valore di depth e il conteggio a 1
                            chunk_dfg[key] = depth_value
                            count_dfg[key] = 1  # Inizializza il conteggio a 1
                        
            else:     
                raise Exception("mode_score wrong") 
            print(chunk_dfg)
            print("------------------")
            print(count_dfg)
            return chunk_dfg
     

    # Get all unique case names
    cases = log["case:concept:name"].unique()
    print("Remaining paths:", len(cases))
    if len(cases) == 0:
       raise Exception("There is no paths with the current value of perc_var and decimal_threshold! Try one less restrictive.") 

    # If n_jobs is -1, use all available CPUs, otherwise use the provided n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count()  # Get the number of available CPU cores
    
    # Ensure n_jobs is at least 1 and no larger than the number of cases
    n_jobs = max(min(n_jobs, len(cases)), 1)  # Ensure n_jobs is within valid range

    # Calculate chunk size, ensure chunk size is at least 1
    chunk_size = max(len(cases) // n_jobs, 1)  # Ensure chunk_size is at least 1
    
    # Split the cases into chunks
    chunks = [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]

    # Process each chunk in parallel
    print("Traversing...")
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk, mode_score) for chunk in chunks)

    # Merge all chunk DFGs into a single DFG dictionary
    print("Aggregating...")
    dfg = {}
    for result in results:
        for key, value in result.items():
            if key in dfg:
                dfg[key] += value  # Aggregate counts for shared transitions
            else:
                dfg[key] = value
    # Return the final DFG dictionary
    return dfg, log

def generate_dot(dfg):
    """
    Creates a Graphviz directed graph (digraph) from a Data Flow Graph (DFG) dictionary and returns the dot representation.

    Args:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.

    Returns:
    dot: A Graphviz dot object representing the directed graph.
    """

    # Initialize a Graphviz digraph with specified attributes
    dot = graphviz.Digraph(
        "dpg",
        engine="dot",
        graph_attr={
            "bgcolor": "white",
            "rankdir": "R",
            "overlap": "false",
            "fontsize": "20",
        },
        node_attr={"shape": "box"},
    )

    # Keep track of added nodes to avoid duplicates
    added_nodes = set()
    
    # Sort the DFG dictionary by values (transition counts) for deterministic order
    sorted_dict_values = {k: v for k, v in sorted(dfg.items(), key=lambda item: item[1])}

    # Iterate through the sorted DFG dictionary
    for k, v in sorted_dict_values.items():
        
        # Add the source node to the graph if not already added
        if k[0] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
                label=f"{k[0]}",
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[0])
        
        # Add the destination node to the graph if not already added
        if k[1] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
                label=f"{k[1]}",
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[1])
        
        # Add an edge between the source and destination nodes with the transition count as the label
        dot.edge(
            str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
            str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
            label=str(v),
            penwidth="1",
            fontsize="18"
        )
    
    # Return the Graphviz dot object
    return dot


def calculate_class_boundaries(key, nodes):
    feature_bounds = {}
    boundaries = []

    for node in nodes:
        parts = re.split(' <= | > ', node)
        feature = parts[0]
        value = float(parts[1])
        condition = '>' in node

        if feature not in feature_bounds:
            feature_bounds[feature] = [math.inf, -math.inf]

        if condition:  # '>' condition
            if value < feature_bounds[feature][0]:
                feature_bounds[feature][0] = value
        else:  # '<=' condition
            if value > feature_bounds[feature][1]:
                feature_bounds[feature][1] = value

    for feature, (min_greater, max_lessequal) in feature_bounds.items():
        if min_greater == math.inf:
            boundary = f"{feature} <= {max_lessequal}"
        elif max_lessequal == -math.inf:
            boundary = f"{feature} > {min_greater}"
        else:
            boundary = f"{min_greater} < {feature} <= {max_lessequal}"
        boundaries.append(boundary)

    return key, boundaries

def calculate_boundaries(class_dict):
    # Using joblib's Parallel and delayed
    results = Parallel(n_jobs=-1)(delayed(calculate_class_boundaries)(key, nodes) for key, nodes in class_dict.items())
    boundaries_class = dict(results)
    return boundaries_class
    


def get_dpg_metrics(dpg_model, nodes_list, outliers_df, event_log, edges_label, log_base, mode, global_bounds):
    """
    Extracts metrics from a DPG.

    Args:
    dpg_model: A NetworkX graph representing the directed process graph.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    data: A dictionary containing the communities and class bounds extracted from the DPG model.
    """
    # Set the random seed for reproducibility
    np.random.seed(42)

    print("Calculating metrics...")
    # Create a dictionary to map node labels to their identifiers
    diz_nodes = {node[1] if "->" not in node[0] else None: node[0] for node in nodes_list}
    # Remove any None keys from the dictionary
    diz_nodes = {k: v for k, v in diz_nodes.items() if k is not None}
    # Create a reversed dictionary to map node identifiers to their labels
    diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}
    
    # Extract asynchronous label propagation communities
    asyn_lpa_communities = nx.community.asyn_lpa_communities(dpg_model, weight='weight', seed=42)
    asyn_lpa_communities_stack = [{diz_nodes_reversed[str(node)] for node in community} for community in asyn_lpa_communities]

    filtered_nodes = {k: v for k, v in diz_nodes.items() if 'Class' in k or 'Pred' in k}
    # Initialize the predecessors dictionary
    predecessors = {k: [] for k in filtered_nodes}
    # Find predecessors using more efficient NetworkX capabilities
    for key_1, value_1 in filtered_nodes.items():
        # Using single-source shortest path to find all nodes with paths to value_1
        # This function returns a dictionary of shortest paths to value_1
        try:
            preds = nx.single_source_shortest_path(dpg_model.reverse(), value_1)
            predecessors[key_1] = [k for k, v in diz_nodes.items() if v in preds and k != key_1]
        except nx.NetworkXNoPath:
            continue    

    # Calculate the class boundaries
    print("Calculating constraints...")
#   class_bounds = calculate_boundaries(predecessors)
    
#EDIT -------------------------------


    paths = extract_paths(event_log)
    inliers_bounds = inliers_class_bounds(paths)
#   outlier_best_paths = best_paths_per_sample(paths)
    
#   formatted_paths = format_extract_paths(paths)
#   formatted_class_bounds = format_class_bounds(class_bounds)
#   formatted_outlier_best_paths = format_best_paths_per_sample(outlier_best_paths)

    
    # Create a data dictionary to store the extracted metrics
    data = {
        "Outliers": outliers_df,
        "Communities": asyn_lpa_communities_stack,
        "Paths": paths,
#       "\nBest Paths": formatted_outlier_best_paths,     
    }
    
    # Add "Anomaly Bounds" to data only if mode is not "global"
    if mode == "local_outliers":
        data["\nInliers Bounds"] = global_bounds
        data["\nOutlier Bounds"] = outlier_class_bounds(paths)
        data["\nAnomaly Bounds"] = verifica_bounds(global_bounds, inliers_bounds)
    elif mode == "global_inliers":
        data["\nInliers Bounds"] = inliers_bounds    

    return data, inliers_bounds


def verifica_bounds(global_bounds, local_bounds):
    # Lista per memorizzare eventuali bounds locali che non rispettano i global bounds
    bounds_non_conformi = {}

    # Itera attraverso i bounds locali per verificare rispetto ai bounds globali
    for feature, local in local_bounds.items():
        global_min = global_bounds.get(feature, {}).get('min', float('-inf'))
        global_max = global_bounds.get(feature, {}).get('max', float('inf'))
        
        local_min = local['min']
        local_max = local['max']
        
        # Verifica se i bounds locali rispettano quelli globali
        if local_min < global_min or local_max > global_max:
            bounds_non_conformi[feature] = local  # Restituisci tutto il bound locale se uno dei due non è conforme

    return bounds_non_conformi








sorted_labels = []


def get_dpg_node_metrics(dpg_model, nodes_list):
    """
    Extracts metrics from the nodes of a DPG model.

    Args:
    dpg_model: A NetworkX graph representing the DPG.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    df: A pandas DataFrame containing the metrics for each node in the DPG.
    """

    def key_dict(dict, value):
        key = None
        for k, v in dict.items():
            if v == value:
                key = k
                break
        return key

    # Dictionary from nodes_list
    node_dict = {item[0]: item[1] for item in nodes_list}

    # Calculate the degree of each node
    degree = dict(nx.degree(dpg_model))
    # Calculate the in-degree (number of incoming edges) for each node
    in_nodes = {node: dpg_model.in_degree(node) for node in list(dpg_model.nodes())}
    # Calculate the out-degree (number of outgoing edges) for each node
    out_nodes = {node: dpg_model.out_degree(node) for node in list(dpg_model.nodes())}
    # Calcolare l'in-degree pesato per tutti i nodi
    in_nodes_weight = {node: in_degree for node, in_degree in dpg_model.in_degree(weight='weight')}
    # Calcolare l'out-degree pesato per tutti i nodi
    out_nodes_weight = {node: out_degree for node, out_degree in dpg_model.out_degree(weight='weight')}
    
    # New metrics: calculate the difference of out-degree for each class: Class-1 - Class1
    
    # Create a dictionary to store the node metrics
    data_node = {
        "Node": list(dpg_model.nodes()),
        "Degree": list(degree.values()),                               # Total degree (in-degree + out-degree)
        "In degree nodes": list(in_nodes.values()),                    # Number of incoming edges
        "Out degree nodes": list(out_nodes.values()),                  # Number of outgoing edges
        
        "In Weight": list(in_nodes_weight.values()),
        "Out Weight": list(out_nodes_weight.values()),
        "Diff": [in_nodes_weight[node] - out_nodes_weight[node] for node in dpg_model.nodes()],  # Difference between in-weight and out-weight   
        "To Inliers": [dpg_model[node][key_dict(node_dict, 'Class 1')]['weight'] if key_dict(node_dict, 'Class 1') in dpg_model[node] else 0 for node in dpg_model],
        "To Outliers": [dpg_model[node][key_dict(node_dict, 'Class -1')]['weight'] if key_dict(node_dict, 'Class -1') in dpg_model[node] else 0 for node in dpg_model],

    }

    # Merge the node metrics with the node labels
    # Assuming data_node and nodes_list are your input data sets
    df_data_node = pd.DataFrame(data_node).set_index('Node')
    df_data_node['In Out'] = df_data_node['To Inliers'] - df_data_node['To Outliers']
    df_data_node['To Inliers Weight'] = df_data_node['To Inliers'] / df_data_node['In Weight']
    df_data_node['To Outliers Weight'] = df_data_node['To Outliers'] / df_data_node['In Weight']
    df_data_node['In Out Weight'] = df_data_node['To Inliers Weight'] - df_data_node['To Outliers Weight']
    df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index('Node')
    df = pd.concat([df_data_node, df_nodes_list], axis=1, join='inner').reset_index()
    
    
    # Sort by 'Diff' in descending order
    df_sorted = df.sort_values(by="Diff", ascending=True)
    
    # Extract the labels from the sorted DataFrame
    sorted_labels_current = df_sorted["Label"].tolist()
    
    # Accumula la lista corrente in sorted_labels globale
    sorted_labels.append(sorted_labels_current)
    
    # Return the resulting DataFrame
    return df



def get_dpg(X_train, feature_names, model, decimal_threshold, predicates, mode_graph, mode_score, n_samples, n_inliers, n_outliers, mode, perc_var=0.0000000000000000000000000000001, n_jobs=-1):
    """
    Generates a DPG from training data and a random forest model.

    Args:
    X_train: A numpy array or similar structure containing the training data samples.
    feature_names: A list of feature names corresponding to the columns in X_train.
    model: A trained random forest model.
    perc_var: A float representing the minimum percentage of total traces a variant must have to be kept. 
    decimal_threshold: The number of decimal places to which thresholds are rounded.
    n_jobs: Number of parallel jobs to run. Default is -1 (use all available CPUs).

    Returns:
    dot: A Graphviz Digraph object representing the DPG.
    event_log: A flattened list of all event logs from the samples.
    """

    print("\nStarting DPG extraction *****************************************")
    print("Model Class:", model.__class__.__name__)
    print("Model Class Module:", model.__class__.__module__)
    print("Model Estimators: ", len(model.estimators_))
    print("Model Params: ", model.get_params())
    print("*****************************************************************")
    max_depth = math.ceil(math.log2(min(256, n_samples)))
    print("Max depth: ", max_depth)

    def process_sample(i, sample):
        """Process a single sample."""
        event_log = tracing_if(i, sample, model, feature_names, decimal_threshold, mode_graph, max_depth, mode)        
        return event_log

    print('Tracing ensemble...')
    logs = Parallel(n_jobs=n_jobs)(
        delayed(process_sample)(i, sample) for i, sample in enumerate(X_train)
    )

    # Flatten the list of lists
    event_log = [item for sublist in logs for item in sublist]
    
    log_df = pd.DataFrame(event_log, columns=["case:concept:name", "concept:name"])
    # Rimuove tutte le righe che contengono "Class -1" o "Class 1"
    log_df_new = log_df[~log_df["concept:name"].str.contains("Class -1|Class 1", na=False)].copy()
    print(f'Total of paths: {len(log_df["case:concept:name"].unique())}')
    with open("log_df.txt", "w") as f:
        f.write(log_df_new.to_string(index=True))
    
    print(f'Filtering structure... (perc_var={perc_var})')
    # Filter the log based on the variant percentage if specified
    filtered_log = log_df
    if perc_var > 0:
        filtered_log = filter_log(log_df, perc_var)
        
    
    # print('Building DPG...')
    # # Discover the Data Flow Graph (DFG) from the filtered log
    dfg, log_base = discover_dfg(filtered_log, predicates, mode_score, max_depth, n_inliers, n_outliers)

    # print('Extracting graph...')
    # # Create a Graphviz Digraph object from the DFG
    dot = generate_dot(dfg)

    # Return both the Graphviz Digraph object and the full event log
    return dot, event_log, log_base



#EDIT -----------------------------------------------



def inliers_class_bounds(paths):

    pattern = r"([a-zA-Z]+[a-zA-Z]*Cm) *(<=|>) *(-?\d+\.?\d*)"
    
    # Dizionario per memorizzare i minimi e massimi per ogni feature
    bounds = defaultdict(lambda: {'min': float('-inf'), 'max': float('inf')})

    # Itera su ogni path e estrai le feature e i valori
    for sample, conditions in paths:
        for condition in conditions:
            match = re.search(pattern, condition)
            if match:
                feature = match.group(1)
                operator = match.group(2)
                value = float(match.group(3))
                
                # Aggiorna i bounds della feature
                if operator == '>':
                    # Aggiorna il minimo
                    bounds[feature]['min'] = max(bounds[feature]['min'], value)
                elif operator == '<=':
                    # Aggiorna il massimo
                    bounds[feature]['max'] = min(bounds[feature]['max'], value)

    return bounds






def outlier_class_bounds(paths):

    pattern = r"([a-zA-Z]+[a-zA-Z]*Cm) *(<=|>) *(-?\d+\.?\d*)"
    # Dizionario per memorizzare i minimi e massimi per ogni feature
    bounds = defaultdict(lambda: {'min': float('-inf'), 'max': float('inf')})

    # Itera su ogni path e estrai le feature e i valori
    for sample, conditions in paths:
        for condition in conditions:
            match = re.search(pattern, condition)
            if match:
                feature = match.group(1)
                operator = match.group(2)
                value = float(match.group(3))
                
                # Aggiorna i bounds della feature
                if operator == '>':
                    bounds[feature]['min'] = max(bounds[feature]['min'], value)
                elif operator == '<=':
                    bounds[feature]['max'] = min(bounds[feature]['max'], value)
    
    return bounds 











def format_class_bounds(class_bounds):
    formatted_output = []

    # Funzione per formattare i limiti di una feature
    def format_limits(feature, limits):
        # Gestisci i casi con inf e -inf
        min_value = "∞" if limits['min'] == float('inf') else ("-∞" if limits['min'] == -float('inf') else str(limits['min']))
        max_value = "∞" if limits['max'] == float('inf') else ("-∞" if limits['max'] == -float('inf') else str(limits['max']))
        return f"    {min_value} <= {feature} <= {max_value}"

    # Verifica se il dizionario ha direttamente le feature con i loro limiti
    if class_bounds:
        formatted_output.append("\n=== Feature Bounds ===")
        for feature, limits in class_bounds.items():
            formatted_output.append(format_limits(feature, limits))

    return '\n'.join(formatted_output)




def extract_paths(event_log):
    """
    Estrae i percorsi per ciascun case_id dall'event_log.

    Args:
    event_log: Una lista di tuple dove ogni tupla contiene (case_id, step).

    Returns:
    all_paths: Una lista di tuple (case_id, path), dove il path è una lista di passi e anomaly score.
    """
    
    # Inizializza un dizionario per raggruppare i path per ogni case_id
    from collections import defaultdict
    paths_dict = defaultdict(list)

    # Riempi il dizionario con i percorsi per ogni case_id
    for case_id, step in event_log:
        paths_dict[case_id].append(step)

    # Converti il dizionario in una lista di tuple (case_id, path)
    return list(paths_dict.items())


def format_extract_paths(paths):
    """
    Format paths with their case_id and anomaly score.

    Args:
    all_paths: Lista di tuple (case_id, path), dove il path è una lista di passi e anomaly score.

    Returns:
    result: Una stringa formattata contenente tutti i percorsi con i rispettivi case_id e anomaly score.
    """
    
    # Costruisci e formatta una lista di stringhe per ogni path
    formatted_paths = [
        f"{case_id}: {' ----> '.join(map(str, path[:-1]))} | Anomaly Score: {round(path[-1], 4)}"
        for case_id, path in paths
    ]

    # Unisci tutte le stringhe in una singola stringa
    return "\n".join(formatted_paths)


def best_paths_per_sample(paths):
    """
    Estrae il miglior percorso per ciascun sample, basato sul case_id.
    
    Args:
    paths: Lista di tuple (case_id, step), dove case_id include un suffisso e step è il percorso.

    Returns:
    all_paths: Lista di tuple (case_id_with_suffix, feature, comparator, best_path).
    """
    # Inizializza un dizionario per raggruppare i path per ogni sample (basato sul prefisso del case_id)
    paths_dict = defaultdict(list)

    # Riempi il dizionario con i percorsi per ogni sample
    for case_id, step in paths:
        sample_id, path_suffix = case_id.split('_', 1)  # Estrai il prefisso e il suffisso
        paths_dict[sample_id].append((path_suffix, step))

    outlier_best_paths = []

    # Analizza ogni sample e determina il miglior percorso
    for sample_id, steps_with_suffix in paths_dict.items():
        # Estrai solo i percorsi (step) per cercare la lunghezza minima
        steps = [step for _, step in steps_with_suffix]

        # Trova la lunghezza minima dei percorsi
        min_length = min(len(step) for step in steps)
        
        # Filtra i percorsi con lunghezza minima
        best_paths = [step for step in steps if len(step) == min_length]

        # Variabili per tenere traccia del miglior percorso per ogni feature
        feature_paths = {}

        # Confronta i percorsi con la stessa lunghezza minima
        for path in best_paths:
            *conditions, class_predicate, anomaly_score = path
            last_condition = conditions[-1]  # Ultima condizione, tipo 'PetalLengthCm <= 28.74'
            
            # Usa una regex per separare la variabile, il comparatore e il valore numerico
            match = re.match(r'(\S+)\s*(<=|>=|<|>)\s*(-?\d+(\.\d+)?)', last_condition)
            if match:
                feature, comparator, value = match.group(1), match.group(2), float(match.group(3))

                # Se la feature non è nel dizionario, inizializzala
                if feature not in feature_paths:
                    feature_paths[feature] = {}

                # Aggiorna il miglior percorso per la feature, se necessario
                current_best = feature_paths[feature].get(comparator)
                if current_best is None or \
                   (comparator in ['>', '>='] and value > current_best[0]) or \
                   (comparator in ['<', '<='] and value < current_best[0]):
                    feature_paths[feature][comparator] = (value, path)

        # Aggiungi i migliori percorsi per ogni feature e comparatore
        for feature, comparators in feature_paths.items():
            for comparator, (value, best_path) in comparators.items():
                # Ripristina il case_id completo con il suffisso
                for suffix, step in steps_with_suffix:
                    if step == best_path:
                        case_id_with_suffix = f"{sample_id}_{suffix}"
                        outlier_best_paths.append((case_id_with_suffix, feature, comparator, best_path))
                        break

    return outlier_best_paths


def format_best_paths_per_sample(outlier_best_paths):
    """
    Formatta i migliori percorsi con case_id, condizioni e anomaly score.

    Args:
    all_paths: Lista di tuple (case_id, feature, comparator, path), dove il path contiene i passi e anomaly score.

    Returns:
    Una stringa formattata contenente tutti i percorsi con case_id e anomaly score.
    """
    formatted_outlier_best_paths = [
        f"{sample_id}: {' ----> '.join(path[:-2])} ----> {path[-2]} | Anomaly Score: {path[-1]:.4f}"
        for sample_id, _, _, path in outlier_best_paths
    ]
    return "\n".join(formatted_outlier_best_paths)

























def feature_depth(condition):
        # Se la condizione è "Class -1", ritorna direttamente
        if condition == "Class -1":
            return condition
        
        # Regex per trovare il nome della feature e l'ultimo numero (indicato come "3" nell'esempio)
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)(?:.*?(\d+))?$", condition)
        
        # Se c'è un match, ritorniamo il nome della feature seguito dal numero trovato (se presente)
        if match:
            feature_name = match.group(1)
            depth = match.group(2) if match.group(2) else ""  # Otteniamo il numero, se presente
            return f"{feature_name} {depth}".strip()  # Rimuoviamo gli spazi inutili
        else:
            return condition  # Se non c'è nessun match, ritorna la condizione originale
        
def feature_operator_depth(condition):
        # Regex per trovare il nome della feature, l'operatore (<, >, <=, >=) e il valore numerico (depth)
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)(\s*(<=|>=|<|>))\s*(\d+\.?\d*)\s*(\d+)", condition)
        
        # Se c'è un match, ritorniamo il nome della feature, l'operatore e la depth
        if match:
            feature_name = match.group(1)  # Nome della feature
            operator = match.group(2)  # Operatore (>, <, >=, <=)
            depth = match.group(5)  # Depth (il valore numerico che segue)
            return f"{feature_name} {operator.strip()} {depth}"  # Restituiamo la feature, l'operatore e la depth
        else:
            return condition  # Se non c'è nessun match, ritorna la condizione originale

def feature_sort(condition, feature_count):
        # Se la condizione è "Class -1", ritorna direttamente
        if condition == "Class -1":
            feature_count.clear()  # Resetta il conteggio senza creare un nuovo dizionario
            return condition
        
        # Regex per trovare il nome della feature e l'ultimo numero (indicato come "3" nell'esempio)
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)(?:.*?(\d+))?$", condition)
        
        # Se c'è un match, ritorniamo il nome della feature seguito dal numero trovato (se presente)
        if match:
            feature_name = match.group(1)
            
            # Incrementiamo il conteggio della feature_name
            feature_count[feature_name] += 1
            sort_count = feature_count[feature_name]
            
            # Ritorniamo il feature_name, depth, e il sort_count (quante volte è apparso finora)
            return f"{feature_name} {sort_count}".strip()  # Rimuoviamo gli spazi inutili
        else:
            return condition  # Se non c'è nessun match, ritorna la condizione originale

def feature_operator_sort(condition, feature_operator_count):
        # Se la condizione è "Class -1", ritorna direttamente
        if condition == "Class -1":
            feature_operator_count.clear()  # Resetta il conteggio senza creare un nuovo dizionario
            return condition
        
        # Regex per trovare il nome della feature, l'operatore (<, >, <=, >=) e il valore numerico (depth)
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)(\s*(<=|>=|<|>))\s*(\d+\.?\d*)\s*(\d+)", condition)
        
        # Se c'è un match, ritorniamo il nome della feature, l'operatore e la depth
        if match:
            feature_name = match.group(1)  # Nome della feature
            operator = match.group(2).strip()  # Operatore (>, <, >=, <=)
            
            # Creiamo una chiave basata su feature_name e operator per contare quante volte appare
            key = f"{feature_name} {operator}"
            feature_operator_count[key] += 1
            sort_count = feature_operator_count[key]
            
            # Ritorniamo la feature_name, l'operatore, la depth e il sort_count
            return f"{feature_name} {operator} {sort_count}"
        else:
            return condition  # Se non c'è nessun match, ritorna la condizione originale
        
        
def get_dpg_edge_metrics(dpg_model, nodes_list):
    """
    Extracts metrics from the edges of a DPG model, including:
    - Edge Load Centrality
    - Trophic Differences
    
    Args:
    dpg_model: A NetworkX graph representing the DPG.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    df: A pandas DataFrame containing the metrics for each edge in the DPG.
    """
    

    # Calculate edge weights (assuming edges have 'weight' attribute)
    edge_weights = nx.get_edge_attributes(dpg_model, 'weight')
    
    # Aggiungi le etichette dei nodi
    edge_data_with_labels = []
    for u, v in dpg_model.edges():
        # Ottieni le etichette per i nodi coinvolti nell'arco
        u_label = next((label for node, label in nodes_list if node == u), None)
        v_label = next((label for node, label in nodes_list if node == v), None)
        
        # Ottieni gli identificativi (ID) per i nodi coinvolti nell'arco
        u_id = next((node for node, label in nodes_list if node == u), None)
        v_id = next((node for node, label in nodes_list if node == v), None)
        
        # Aggiungi i dati per l'arco con le etichette e gli ID
        edge_data_with_labels.append([f"{u}-{v}",  
                                     edge_weights.get((u, v), 0),
                                     u_label, v_label, u_id, v_id])
    
    # Crea un DataFrame con gli archi, le etichette e gli ID
    df_edges_with_labels = pd.DataFrame(edge_data_with_labels, columns=["Edge", "Weight", 
                                                                        "Node_u_label", "Node_v_label", "Source_id", "Target_id"])
    

    # Restituisci il DataFrame risultante
    return df_edges_with_labels