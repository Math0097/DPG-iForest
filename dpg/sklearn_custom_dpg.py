import pandas as pd
import numpy as np
import ntpath
import os
import re
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import matplotlib.pyplot as plt
import portion as P

from .core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics, get_dpg_edge_metrics
from .visualizer import plot_dpg


def select_custom_dataset(path):
    """
    Loads a custom dataset from a CSV file, separates the target column, and prepares the data for modeling.

    Args:
    path: The file path to the CSV dataset.

    Returns:
    data: A numpy array containing the feature data.
    features: A numpy array containing the feature names.
    target: A numpy array containing the target variable.
    """
    # Load the dataset from the specified CSV file
    df = pd.read_csv(path, sep=',')
    
    
    # Convert the feature data to a numpy array
    data = []
    for index, row in df.iterrows():
        data.append([row[j] for j in df.columns])
    data = np.array(data)
    
    # Extract feature names
    features = np.array([i for i in df.columns])

    # Return the feature data, feature names, and target variable
    return data, features



def test_base_sklearn(datasets, n_learners, decimal_threshold, contamination, seed, file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, class_flag=False, predicates="feature_operator", mode="global_inliers", mode_graph="all", mode_score="log2", edge_attribute=None):     #EDITED
    """
    Trains a Random Forest classifier on a selected dataset, evaluates its performance, and optionally plots the DPG.

    Args:
    datasets: The path to the custom dataset to use.

    n_learners: The number of trees in the Random Forest.
    decimal_threshold: Decimal precision of each feature.
    contamination: Rate of outliers.    #EDITED
    file_name: The name of the file to save the evaluation results. If None, prints the results to the console.
    plot: Boolean indicating whether to plot the DPG. Default is False.
    save_plot_dir: Directory to save the plot image. Default is "examples/".
    attribute: A specific node attribute to visualize. Default is None.
    communities: Boolean indicating whether to visualize communities. Default is False.
    class_flag: Boolean indicating whether to highlight class nodes. Default is False.


    Returns:
    df: A pandas DataFrame containing node metrics.
    df_dpg: A pandas DataFrame containing DPG metrics.
    """
    
    # Load dataset
    data, features_name = select_custom_dataset(datasets)
    n_samples = data.shape[0]  # Numero di campioni (righe)
    
    iForest = IsolationForest(n_estimators=n_learners, random_state=seed, contamination=contamination)  
    iForest.fit(data)
    predictions = iForest.predict(data)

    inliers = data[predictions == 1]
    outliers = data[predictions == -1]
    
    # Contiamo il numero di inliers (previsione == 1)
    n_inliers = np.sum(predictions == 1)

    # Contiamo il numero di outliers (previsione == -1)
    n_outliers = np.sum(predictions == -1)

    # Converti in un DataFrame e assegna i nomi delle feature come colonne
    outliers_df = pd.DataFrame(outliers, columns=features_name)
    print(outliers_df)
    
    # Converti gli inliers in un DataFrame
    inliers_df = pd.DataFrame(inliers, columns=features_name)
    
    # Combinare inliers e outliers in un unico DataFrame
    combined_df = pd.concat([inliers_df, outliers_df], ignore_index=True)

    # Calcola i minimi e massimi dagli inliers del dataset
    inliers_min_max = {}
    for feature in inliers_df.columns:
        feature_values = inliers_df[feature]
        feature_min = feature_values.min()
        feature_max = feature_values.max()

        # Aggiungi i limiti per ciascuna feature
        inliers_min_max[feature] = {
            '>': feature_min,
            '<=': feature_max
        }
        
    # Calcola i minimi e massimi tra tutto il dataset
    dataset_min_max = {}
    for feature in combined_df.columns:
        feature_values = combined_df[feature]
        feature_min = feature_values.min()
        feature_max = feature_values.max()

        # Aggiungi i limiti per ciascuna feature
        dataset_min_max[feature] = {
            '>': feature_min,
            '<=': feature_max
        }

    anomaly_bounds = {}
    
    if(mode == "global"):         
        # Extract DPG
        dot, event_log, log_base = get_dpg(data, features_name, iForest, decimal_threshold, predicates, mode_graph, mode_score, n_samples, n_inliers, n_outliers, mode)  
        
        print(log_base.tail(40))
        paths = extract_paths(event_log)
                
        # Convert Graphviz Digraph to NetworkX DiGraph  
        dpg_model, nodes_list, edges_label = digraph_to_nx(dot)

        if len(nodes_list) < 2:
            raise Exception("Warning: Less than two nodes resulted.")
                      
        # Get metrics from the DPG
        df_nodes = get_dpg_node_metrics(dpg_model, nodes_list)
        df_edges = get_dpg_edge_metrics(dpg_model, nodes_list)
        df_dpg = get_dpg_metrics(dpg_model, nodes_list, outliers_df, event_log, edges_label, log_base, mode, paths, global_bounds={}, local_bounds={}, anomaly_bounds={})
                
        # Plot the DPG if requested
        if plot:
            plot_name = (
                    os.path.splitext(ntpath.basename(datasets))[0]
                    + "_"
                    + "iForest"
                    + "_bl"
                    + str(n_learners)
                    + "_dec"
                    + str(decimal_threshold)
                    + "_"
                    + str(predicates)
                    + "_"
                    + str(mode_graph)
                    + "_"
                    + str(mode_score)
            )

            plot_dpg(
                    plot_name,
                    dot,
                    df_nodes,
                    df_edges,
                    df_dpg,
                    save_dir=save_plot_dir,
                    attribute=attribute,
                    communities=communities,
                    class_flag=class_flag,
                    edge_attribute=edge_attribute
            )
        return df_nodes, df_dpg, -1, df_edges
                    
    elif mode== "local_outliers":
                
        # estraggo i bounds degli inliers per ottenere poi gli anomaly bounds
        dot, event_log, log_base = get_dpg(inliers, features_name, iForest, decimal_threshold, predicates, mode_graph, mode_score, n_samples, n_inliers, n_outliers, mode)
      
        paths_in = extract_paths(event_log)

        # Liste per accumulare i risultati di ogni iterazione
        df_list = []
        df_dpg_list = []
        index_list = []
        df_edges_list = []
        local_bounds_list = []

        for index, row in outliers_df.iterrows():
            # Estrai i dati della riga come un array numpy o una lista
            data_sample = row[features_name].values.reshape(1, -1)  # o .tolist() se la funzione richiede una lista    
                
            outlier = pd.DataFrame([row], columns=features_name)       
                
            # Extract DPG
            dot, event_log, log_base = get_dpg(data_sample, features_name, iForest, decimal_threshold, predicates, mode_graph, mode_score, n_samples, n_inliers, n_outliers, mode)
                
            # Convert Graphviz Digraph to NetworkX DiGraph  
            dpg_model, nodes_list, edges_label = digraph_to_nx(dot)
            
            paths = extract_paths(event_log)
            local_bounds = outlier_class_bounds(paths)
            local_bounds_list.append(local_bounds)
            #global_bounds = aggiorna_bounds(global_bounds, local_bounds)
            #anomaly_bounds = verifica_bounds(global_bounds, local_bounds)

            if len(nodes_list) < 2:
                raise Exception("Warning: Less than two nodes resulted.")
                    
            # Get metrics from the DPG
            df_nodes = get_dpg_node_metrics(dpg_model, nodes_list)
            df_edges = get_dpg_edge_metrics(dpg_model, nodes_list)
            df_dpg = get_dpg_metrics(dpg_model, nodes_list, outlier, event_log, edges_label, log_base, mode, paths, {}, local_bounds, anomaly_bounds)
                
            # Aggiungi i risultati alle liste
            df_list.append(df_nodes)
            df_dpg_list.append(df_dpg)
            index_list.append(index)
            df_edges_list.append(df_edges)
                

            # Plot the DPG if requested
            if plot:
                plot_name = (
                        os.path.splitext(ntpath.basename(datasets))[0]
                        + "_"
                        + "iForest"
                        + "_bl"
                        + str(n_learners)
                        + "_dec"
                        + str(decimal_threshold)
                        + "_"
                        + str(predicates)
                        + "_anomaly"
                        + str(index)
                        + "_"
                        + str(mode_graph)
                        + "_"
                        + str(mode_score)
                )

                plot_dpg(
                        plot_name,
                        dot,
                        df_nodes,
                        df_edges, #EDIT
                        df_dpg,
                        save_dir=save_plot_dir,
                        attribute=attribute,
                        communities=communities,
                        class_flag=class_flag,
                        edge_attribute=edge_attribute,
                )

        print(outliers_df)
        
        global_bounds = inliers_class_bounds(paths_in)
        #print("Global:", global_bounds)
        print("Local bounds:", local_bounds_list)
        feature_vectors = define_feature_vector(local_bounds_list, outliers_df, inliers_min_max)
        print(feature_vectors)
        plot_features_bounds(local_bounds_list, dataset_min_max, outliers_df)
        #final_bounds = define_bounds(inliers_min_max, local_bounds_list)
        #print("Final bounds:", new_bounds)
        #intervals_to_plot = generate_intervals(final_bounds)

        #print("Intervals:", intervals)
        #plot_intervals(intervals_to_plot)
          
        # Restituisci le liste con tutti i risultati
        return df_list, df_dpg_list, index_list, df_edges_list


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
    
    
def inliers_class_bounds(paths):
    pattern = r"(\S+)\s*(<=|<|>|>=)\s*(-?[\d.]+)"
    
    with open("global_paths.txt", 'w') as f:
        for sample, conditions in paths:
            f.write(f"Sample: {sample}\n")
            for condition in conditions:
                f.write(f"{condition}\n")
            f.write("\n") 
    
    # Dizionario per memorizzare i minimi e massimi per ogni feature
    bounds = defaultdict(lambda: {'>': float("inf"), '<=': float("-inf")})

    # Itera su ogni path e estrai le feature e i valori
    for sample, conditions in paths:
        for condition in conditions:

            match = re.search(pattern, condition)
            if match:
                feature = match.group(1)
                operator = match.group(2)
                value = float(match.group(3))

                # Se l'operatore è ">", aggiorna il bound "<="
                if operator == '>':
                    bounds[feature]['<='] = max(bounds[feature]['<='], value)
                
                # Se l'operatore è "<=", aggiorna il bound ">"
                elif operator == '<=':
                    bounds[feature]['>'] = min(bounds[feature]['>'], value)  
                        
    bounds = dict(bounds)
    return bounds


def outlier_class_bounds(paths): # mettere last_condition e rimuovere commenti per avere solo ultima condizione
    pattern = r"(\S+)\s*(<=|<|>|>=)\s*(-?[\d.]+)"
    
    # Dizionario per memorizzare i minimi e massimi per ogni feature
    bounds = defaultdict(lambda: {'>': float("inf"), '<=': float("-inf")})

    # Itera su ogni path e estrai le feature e i valori
    for sample, conditions in paths:
        last_condition = None
        
        # Trova l'ultima condizione prima di "Class -1"
        for condition in conditions:
            if "Class -1" in condition:
                break
            last_condition = condition
        
        # Se abbiamo trovato una condizione valida prima di "Class -1"
        if last_condition:
            match = re.search(pattern, last_condition)
            if match:
                feature = match.group(1)
                operator = match.group(2)
                value = float(match.group(3))

                # Condizione per l'operatore ">"
                if operator == '>':
                    # Aggiorna solo se il valore è maggiore di quello nel global_bounds
                    #if value > global_bounds.get(feature, {}).get('<=', float('inf')):
                    bounds[feature]['>'] = min(bounds[feature]['>'], value)
                
                # Condizione per l'operatore "<="
                elif operator == '<=':
                    # Aggiorna solo se il valore è minore di quello nel global_bounds
                    #if value < global_bounds.get(feature, {}).get('>', float('-inf')):
                    bounds[feature]['<='] = max(bounds[feature]['<='], value)    
   
   
    # Verifica e inverte i bounds con inf o -inf
    for feature, feature_bounds in bounds.items():
        for operator in ['>', '<=']:
            if feature_bounds[operator] == float('inf'):
                # Se il bound è inf, lo inverto in -inf
                feature_bounds[operator] = float('-inf')
            elif feature_bounds[operator] == float('-inf'):
                # Se il bound è -inf, lo inverto in inf
                feature_bounds[operator] = float('inf')   

    bounds = dict(bounds)
    return bounds


def verifica_bounds(global_bounds, local_bounds):
    # Lista per memorizzare eventuali bounds locali che non rispettano i global bounds
    bounds_non_conformi = {}

    # Itera attraverso i bounds locali per verificare rispetto ai bounds globali
    for feature, local in local_bounds.items():
        global_min = global_bounds.get(feature, {}).get('>', float('-inf'))
        global_max = global_bounds.get(feature, {}).get('<=', float('inf'))
        
        local_min = local['>']
        local_max = local['<=']
        
        # Verifica se il limite inferiore locale è maggiore del limite superiore globale
        # o se il limite superiore locale è minore del limite inferiore globale
        if local_min > global_max or local_max < global_min:
            bounds_non_conformi[feature] = local  # Se uno dei due è fuori, è un bound anomalo

    return bounds_non_conformi


def aggiorna_bounds(global_bounds, local_bounds): #non può funzionare, mi mostra solo i bounds tenendo conto delle anomlie più esterne e non anche quelle più interne
    for key, local in local_bounds.items():
        # Verifica se il key esiste nei global bounds
        if key in global_bounds:
            global_bound = global_bounds[key]

            # Aggiorna limite inferiore se è definito nel local bounds e non è -inf
            if '>' in local and local['>'] != float('inf'):
                global_bound['<='] = local['>']

            # Aggiorna limite superiore se è definito nel local bounds e non è inf
            if '<=' in local and local['<='] != float('-inf'):
                global_bound['>'] = local['<=']
    
    return global_bounds


def define_bounds(global_bounds, local_bounds_list):
    
    intervals_global = {}
    for feature, bounds in global_bounds.items():
        # Intervallo aperto a sinistra e chiuso a destra
        interval = P.open(bounds['>'], bounds['<='])
        intervals_global[feature] = interval
    
    
    def create_interval(bounds):
        # Intervallo aperto a sinistra e chiuso a destra
        lower_bound = bounds['>']
        upper_bound = bounds['<=']
        return P.open(lower_bound, upper_bound)


    all_intervals_local = []
    for item in local_bounds_list:
        intervals = {}
        for feature, bounds in item.items():
            intervals[feature] = create_interval(bounds)
        all_intervals_local.append(intervals)
    
    unified_intervals = {}
    for intervals in all_intervals_local:
        for feature, interval in intervals.items():
            if feature in unified_intervals:
                unified_intervals[feature] |= interval  # Unione degli intervalli
            else:
                unified_intervals[feature] = interval
    
    
    # Calcolo della differenza tra l'intervallo globale e l'unione degli intervalli locali
    difference_intervals = {}
    for feature, global_interval in intervals_global.items():
        if feature in unified_intervals:
            # Calcola la differenza tra globale e unificato
            difference_intervals[feature] = global_interval - unified_intervals[feature]
        else:
            # Se non c'è intervallo unificato per la feature, la differenza è il globale stesso
            difference_intervals[feature] = global_interval

    return difference_intervals


def generate_intervals(difference_intervals):
    """
    Genera combinazioni di intervalli per ciascuna feature in modo dinamico,
    supportando un numero arbitrario di intervalli.

    Args:
    - difference_intervals (dict): Dizionario contenente gli intervalli delle feature
      (es. `portion.Interval`).

    Returns:
    - list: Una lista di combinazioni di intervalli sotto forma di dizionari.
    """
    # Estrai i nomi delle feature
    feature_names = list(difference_intervals.keys())
    feature_x = feature_names[0]  # Prima feature (asse X)
    feature_y = feature_names[1]  # Seconda feature (asse Y)
    
    # Calcola i complementari rispetto a (-∞, +∞)
    global_domain = P.open(-float('inf'), float('inf'))
    complementary_intervals = {
        feature: global_domain - intervals
        for feature, intervals in difference_intervals.items()
    }

    # Converti gli intervalli in liste di sotto-intervalli
    x_intervals = list(difference_intervals[feature_x])
    y_intervals = list(difference_intervals[feature_y])

    # Genera combinazioni
    intervals = []

    # Combina ogni intervallo di X con tutti gli intervalli di Y
    for x_range in x_intervals:
        for y_range in y_intervals:
            intervals.append({
                feature_x: (x_range.lower, x_range.upper),
                feature_y: (y_range.lower, y_range.upper)
            })
    return intervals


def plot_intervals(intervals):
    """
    Plotta gli intervalli dati in un grafico 2D utilizzando tuple per rappresentare gli intervalli.

    Args:
    - intervals (list): Lista di dizionari con chiavi dinamiche (nomi delle feature),
      contenenti intervalli rappresentati come tuple `(lower, upper)`.
    """
    plt.figure(figsize=(10, 6))

    # Controllo per intervalli vuoti
    if not intervals:
        print("Nessun intervallo da plottare.")
        return

    # Prendi i nomi dinamici delle feature dal primo intervallo
    feature_x = list(intervals[0].keys())[0]  # Nome della prima feature (X-axis)
    feature_y = list(intervals[0].keys())[1]  # Nome della seconda feature (Y-axis)

    # Plot di ogni intervallo con lo stesso colore
    for i, interval in enumerate(intervals):
        x_range = interval[feature_x]
        y_range = interval[feature_y]

        # Assumiamo che x_range e y_range siano tuple (lower, upper)
        if x_range[0] is not None and x_range[1] is not None:
            if y_range[0] is not None and y_range[1] is not None:
                plt.fill_betweenx(
                    y=[y_range[0], y_range[1]],
                    x1=x_range[0],
                    x2=x_range[1],
                    alpha=0.4,
                    color='blue'  # Colore fisso per tutte le aree
                )

    # Configurazione del grafico
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.grid(True)
    plt.title("Grafico degli Intervalli")
    plt.show()
    
    
  
  
  
def define_feature_vector(local_bounds_list, outliers_df, global_bounds):
    
    feature_vector = {}
    
    # Itera su ogni anomalia nella lista e su ogni riga nel DataFrame degli outliers
    for anomaly_idx, (anomaly_bounds, outlier_row) in enumerate(zip(local_bounds_list, outliers_df.to_dict(orient='records'))):
        distances = {}  # Per salvare le distanze normalizzate per ciascuna feature in questa anomalia
        
        for feature, bounds in anomaly_bounds.items():
            min_all = global_bounds[feature]['>']
            max_all = global_bounds[feature]['<=']
            
            min_bound = bounds['>']  # Limite inferiore
            max_bound = bounds['<=']  # Limite superiore
            outlier_value = outlier_row[feature]  # Valore dell'anomalia
            
            # Calcola il range della feature
            feature_range = max_all - min_all
            
            # Evita divisioni per zero se il range è nullo
            if feature_range == 0:
                print(f"Warning: Feature '{feature}' has a range of 0. Skipping normalization.")
                continue
            
            # Calcola le distanze dal valore dell'anomalia ai limiti
            distance_to_min = abs(outlier_value - min_bound)
            distance_to_max = abs(outlier_value - max_bound)
            
            # Somma le distanze e normalizza
            total_distance = distance_to_min + distance_to_max
            normalized_distance = total_distance / feature_range  # Normalizzazione
            
            distances[feature] = normalized_distance  # Salva la distanza normalizzata per la feature
        
        # Ordina le feature per distanza normalizzata decrescente
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=True))
        
        # Aggiungi il vettore di distanze ordinate per l'anomalia corrente
        feature_vector[f"Anomaly_{anomaly_idx}"] = sorted_distances
    
    return feature_vector
   
   
   
   
    
    
def plot_features_bounds(features_bounds_list, global_bounds_dataset, outliers_df):
    # Itera attraverso le feature globali
    for i, (anomaly, outlier) in enumerate(zip(features_bounds_list, outliers_df.to_dict(orient='records'))):
        # Crea una finestra con tante sottotrame quante sono le feature
        num_features = len(anomaly.keys())
        fig, axes = plt.subplots(num_features, 1, figsize=(10, 3 * num_features), constrained_layout=True)
        
        if num_features == 1:
            axes = [axes]  # Assicura che 'axes' sia sempre una lista
        
        for ax, (feature, bounds) in zip(axes, anomaly.items()):
            min_value = bounds['>']
            max_value = bounds['<=']
            
            # Sostituisci valori infiniti con limiti definiti per il grafico
            min_dataset_all = global_bounds_dataset[feature]['>']
            max_dataset_all = global_bounds_dataset[feature]['<=']
            if min_value == -np.inf:
                min_value = min_dataset_all
            if max_value == np.inf:
                max_value = max_dataset_all

            # Ottieni il valore relativo della feature dall'outlier
            outlier_value = outlier.get(feature, None)
            if outlier_value is not None:
                # Grafico a barre monodimensionale
                ax.barh(feature, max_value - min_value, left=min_value, color='skyblue', height=0.01)
                
                # Aggiungi una croce sul valore relativo dell'outlier
                ax.scatter(outlier_value, feature, color='red', marker='x', s=200, linewidth=2)
                
                # Imposta i limiti dell'asse x
                ax.set_xlim(min_dataset_all, max_dataset_all)
                ax.set_xlabel('Value Range')
                ax.set_title(f"Feature: {feature}")
        
        fig.suptitle(f"Bounds for Features in Anomaly {i}", fontsize=16)
        plt.show()
