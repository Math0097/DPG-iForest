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

from decimal import Decimal, ROUND_HALF_UP

import time

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

        if is_leaf:
            if score < 0 and depth < max_depth: # outlier e non lunghezza max = Class -1
                path.append("Class -1")
            elif score > 0:                     # inlier e qualsiasi lunghezza = Class 1
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

        if score < 0 and sample_path and sample_path[-1] == "Class -1":     # non salva i path degli outlier di lunghezza max
            tree_events = [[f"sample{case_id}_dt{i}", step] for step in sample_path]
            event_log.extend(tree_events)
        elif score > 0 and sample_path and sample_path[-1] == "Class 1":
            tree_events = [[f"sample{case_id}_dt{i}", step] for step in sample_path]
            event_log.extend(tree_events)
    return event_log


def filter_log(log, n_jobs=-1):
    
    """
    Filters a log based on the variant percentage. Variants (unique sequences of activities for cases) 
    that occur less than the specified threshold are removed from the log.

    Args:
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.
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
    
    print('>>>0>>>>>')

    # Split the cases into chunks
    chunks = [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]
    
    # Process each chunk in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)

    # Combine results into a single dictionary
    print('>>>1>>>>>')
    variants = {}
    for result in results:
        for key, value in result.items():
            if key in variants:
                variants[key].extend(value)
            else:
                variants[key] = value

    # Get the total number of unique traces in the log
    total_traces = log["case:concept:name"].nunique()
    print('>>>2>>>>>', total_traces)
    perc_var = 0.0001  # Percentage of variants to keep (1%)
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


def discover_dfg(log, predicates, mode_score, max_depth, n_outliers, n_inliers, n_jobs=-1):

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

    function_dict = {
        'feature': feature,
        'feature_operator': feature_operator,
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
            if mode_score == "freq_pes": #più è piccolo meglio è
                for case in chunk:
                    # Extract the trace (sequence of activities) for the current case
                    trace_df = log[log["case:concept:name"] == case].copy()
                    trace_df.sort_values(by="case:concept:name", inplace=True)

                    # Iterate through the trace to capture transitions between consecutive activities
                    for i in range(len(trace_df) - 1):
                        key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])  # Transition
                        score = Decimal(trace_df.iloc[i, 3]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                        
                        if key in chunk_dfg:
                            chunk_dfg[key] += score  # Increment count if transition exists
                        else:
                            chunk_dfg[key] = score  # Initialize count if transition is new
            
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
                        
            else:     
                raise Exception("mode_score wrong") 
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


def discover_dfg_vero(log, predicates, mode_score, max_depth, n_outliers, n_inliers, n_jobs=-1):
    """
    Estrae nodi e relazioni dal log degli eventi per costruire un Data Flow Graph (DFG)
    che rappresenta la frequenza (ponderata) delle transizioni tra attività.
    
    Argomenti:
      log: DataFrame contenente il log degli eventi. Si assume che contenga almeno le colonne
           'case:concept:name' e 'concept:name'.
      predicates: Indica il tipo di operazione da eseguire sulla colonna 'concept:name'.
                  In questo esempio, se è 'feature_operator' verrà applicata una regex.
      mode_score: Modalità di calcolo del punteggio. Attualmente supportato "freq_pes".
      max_depth: (Non utilizzato in questa versione, ma potrebbe essere utile per eventuali filtri).
      n_outliers: Numero di outlier.
      n_inliers: Numero di inlier.
      n_jobs: Numero di thread da utilizzare. In questa versione vettorializzata non viene usato,
              ma è mantenuto per compatibilità con la firma originale.
              
    Ritorna:
      dfg: Dizionario in cui le chiavi sono tuple (attività corrente, attività successiva) e
           i valori sono la somma ponderata delle transizioni.
      log: DataFrame aggiornato contenente le colonne processate.
    """
    start_time = time.time()

    # --- 1. Preprocessing vettoriale della colonna 'concept:name' ---
    if predicates == 'feature_operator':
        # Definizione della regex
        regex = r"(?P<feature>[A-Za-z][A-Za-z0-9_]*)\s*(?P<operator><=|<|>|>=)\s*(?P<value>-?[\d.]+)\s*(?P<score>-?[\d.]+)\s*(?P<depth>-?[\d.]+)"
        # Applica la regex in modo vettoriale
        extracted = log['concept:name'].str.extract(regex)
        
        # Identifica le righe in cui la regex ha avuto successo
        mask = extracted['feature'].notna()
        
        # Per le righe corrispondenti, aggiorna le colonne con i valori estratti
        log.loc[mask, 'concept:name'] = extracted.loc[mask, 'feature'] + ' ' + extracted.loc[mask, 'operator']
        log.loc[mask, 'concept:name:value'] = pd.to_numeric(extracted.loc[mask, 'value'], errors='coerce')
        log.loc[mask, 'concept:name:score'] = pd.to_numeric(extracted.loc[mask, 'score'], errors='coerce')
        # Per 'depth' se non viene convertito, viene sostituito con 0 (o un altro valore di default se preferisci)
        log.loc[mask, 'concept:name:depth'] = pd.to_numeric(extracted.loc[mask, 'depth'], errors='coerce').fillna(0).astype(int)
        
        # Per le righe che non corrispondono, mantieni il valore originale e imposta score a NaN
        # in modo che vengano successivamente scartate nel calcolo del moltiplicatore.
        log.loc[~mask, 'concept:name:score'] = np.nan
    else:
        # Se si desidera gestire altri predicati, inserirli qui.
        pass

    # --- 2. Ordinamento del log ---
    # Se esiste una colonna 'timestamp' si ordina anche per essa per rispettare l'ordine cronologico.
    if 'timestamp' in log.columns:
        log = log.sort_values(by=["case:concept:name", "timestamp"])
    else:
        log = log.sort_values(by=["case:concept:name"])

    # --- 3. Calcolo del moltiplicatore per ciascun caso ---
    # Per ogni trace (caso) si utilizza il punteggio della prima riga per decidere il moltiplicatore:
    #   - Se il punteggio è 0: multiplier = (n_outliers + n_inliers) / n_outliers
    #   - Se il punteggio è 1: multiplier = (n_outliers + n_inliers) / n_inliers
    #   - Altrimenti il caso viene scartato.
    def compute_multiplier(score_series):
        first_score = score_series.iloc[0]
        if first_score == 0:
            m = (n_outliers + n_inliers) / n_outliers
        elif first_score == 1:
            m = (n_outliers + n_inliers) / n_inliers
        else:
            m = np.nan  # Indica che il caso va ignorato
        return pd.Series(m, index=score_series.index)
    
    log['multiplier'] = log.groupby("case:concept:name")["concept:name:score"].transform(compute_multiplier)
    # Rimuove i casi per cui il moltiplicatore non è definito (ovvero, quelli con punteggio diverso da 0 o 1)
    log = log.dropna(subset=['multiplier'])

    # --- 4. Creazione delle transizioni ---
    # Utilizza il metodo shift per ottenere la transizione verso l'attività successiva per ogni caso
    log['next_concept'] = log.groupby("case:concept:name")["concept:name"].shift(-1)
    # Rimuove le righe in cui non è presente una transizione (ad es. l'ultima riga di ogni caso)
    transitions_df = log.dropna(subset=['next_concept'])

    # --- 5. Aggregazione vettoriale delle transizioni ---
    # Raggruppa per coppia (attività corrente, attività successiva) e somma i moltiplicatori
    dfg_series = transitions_df.groupby(['concept:name', 'next_concept'])['multiplier'].sum()
    dfg = dfg_series.to_dict()

    end_time = time.time()
    print(f"Processing Time: {round(end_time - start_time, 2)} sec")

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
    


def get_dpg_metrics(dpg_model, nodes_list, outliers_df, event_log, edges_label, log_base, mode, paths, global_bounds, local_bounds, anomaly_bounds):
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



    
    # Create a data dictionary to store the extracted metrics
    data = {
        "Outliers": outliers_df,
        "Communities": asyn_lpa_communities_stack,
        "Paths": paths,
        "Global bounds": global_bounds,
        "Local bounds": local_bounds,
        "Anomaly bounds": anomaly_bounds,  
    }

    return data




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
    # Calcolare l'out-degree pesato per tutti i nodi
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
    df_data_node['Inlier-Outlier Propagation Score'] = df_data_node['To Inliers Weight'] - df_data_node['To Outliers Weight']
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



def get_dpg(X_train, feature_names, model, decimal_threshold, predicates, mode_graph, mode_score, 
            n_samples, n_inliers, n_outliers, mode, n_jobs=-1):
    """
    Generates a DPG from training data and a random forest model.
    """
    
    print("\nStarting DPG extraction *****************************************")
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Model Class Module: {model.__class__.__module__}")
    print(f"Model Estimators: {len(model.estimators_)}")
    print(f"Model Params: {model.get_params()}")
    print("*****************************************************************")

    max_depth = math.ceil(math.log2(min(256, n_samples)))
    print(f"Max depth: {max_depth}")

    # ** Funzione ottimizzata per processare i campioni in parallelo **
    def process_sample_batch(batch):
        """Elabora un batch di campioni"""
        return [
            tracing_if(i, sample, model, feature_names, decimal_threshold, mode_graph, 
                       max_depth, mode) 
            for i, sample in batch
        ]

    print("Tracing ensemble...")
    start_time = time.time()

    # ** Dividere X_train in batch per ridurre overhead della parallelizzazione **
    batch_size = max(1, len(X_train) // (4 * n_jobs if n_jobs > 1 else 1))
    batches = [(i, X_train[i]) for i in range(len(X_train))]

    logs = Parallel(n_jobs=n_jobs, batch_size="auto")(
        delayed(process_sample_batch)(batches[i:i + batch_size]) 
        for i in range(0, len(batches), batch_size)
    )

    end_time = time.time()
    print(f"Time: {round(end_time - start_time, 2)} sec")

    # ** Flatten the list of lists **
    event_log = [item for sublist in logs for batch in sublist for item in batch]

    log_df = pd.DataFrame(event_log, columns=["case:concept:name", "concept:name"])
    print(f"Total paths: {len(log_df['case:concept:name'].unique())}")
    print(log_df)
    log_df.to_string("log.txt", index=False)

    #print("Filtering structure...")
    #filtered_log = filter_log(log_df)

    #log_df.to_csv('FULL_log_df.csv', index=False)
    print("Building DPG...")
    dfg, log_base = discover_dfg_vero(log_df, predicates, mode_score, max_depth, n_outliers, n_inliers)

    print("Extracting graph...")
    dot = generate_dot(dfg)

    return dot, event_log, log_base



#EDIT -----------------------------------------------




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
    df_edges_with_labels = pd.DataFrame(edge_data_with_labels, columns=["Edge", "Weighted frequency", 
                                                                        "Node_u_label", "Node_v_label", "Source_id", "Target_id"])
    

    # Restituisci il DataFrame risultante
    return df_edges_with_labels