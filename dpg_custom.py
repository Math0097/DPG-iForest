import os
import argparse

import dpg.sklearn_custom_dpg as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="Basic dataset to be analyzed")

    parser.add_argument("--l", type=int, default=5, help="Number of learners for the Isolation Forest")
    parser.add_argument("--t", type=int, default=2, help="Decimal precision of each feature")
    parser.add_argument("--cont", type=float, default=0.01, help="Rate of outliers")    #EDITED
    parser.add_argument("--seed", type=int, default=42, help="Random seed") #EDITED
    parser.add_argument("--dir", type=str, default="examples/", help="Directory to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the DPG, add the argument to use it as True")
    parser.add_argument("--save_plot_dir", type=str, default="examples/", help="Directory to save the plot image")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize")
    parser.add_argument("--communities", action='store_true', help="Boolean indicating whether to visualize communities, add the argument to use it as True")
    parser.add_argument("--class_flag", action='store_true', help="Boolean indicating whether to highlight class nodes, add the argument to use it as True")
    parser.add_argument("--predicates", type=str, default="feature_operator", help="Type of predicate")    #EDITED
    parser.add_argument("--mode", type=str, default="global_inliers", help="Local/Global (global_outliers global_inliers local_outliers)")    #EDITED
    parser.add_argument("--mode_graph", type=str, default="all", help="All graph or last decisions")    #EDITED
    parser.add_argument("--mode_score", type=str, default="log2", help="Frequency or aggregate score (log2)")    #EDITED
    parser.add_argument("--edge_attribute", type=str, default=None, help="A specific edge attribute to visualize") #EDITED
    args = parser.parse_args()


    df_node_list, df_dpg_metrics_list, index_list, df_edge_list = test.test_base_sklearn(datasets = args.ds,

                                            n_learners = args.l, 
                                            decimal_threshold = args.t,
                                            contamination = args.cont,  #EDITED
                                            seed = args.seed,
                                            file_name = os.path.join(args.dir, f'custom_l{args.l}_t{args.t}_stats.txt'),
                                            plot = args.plot, 
                                            save_plot_dir = args.save_plot_dir, 
                                            attribute = args.attribute, 
                                            communities = args.communities, 
                                            class_flag = args.class_flag,
                                            predicates = args.predicates,
                                            mode = args.mode,
                                            mode_graph = args.mode_graph,
                                            mode_score = args.mode_score,
                                            edge_attribute = args.edge_attribute)

    
    nameDataset = os.path.splitext(os.path.basename(args.ds))[0]
    
    if(args.mode == "global" or args.mode == "global_outliers"):
        df_edge_list.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_{args.mode_graph}_{args.mode_score}_global_edge_metrics.csv'),
                    encoding='utf-8')
        
        df_node_list.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_{args.mode_graph}_{args.mode_score}_global_node_metrics.csv'),
                    encoding='utf-8')

        with open(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_{args.mode_graph}_{args.mode_score}_global_dpg_metrics.txt'), 'w') as f:
            for key, value in df_dpg_metrics_list.items():
                # Controlla se il valore è una tupla
                if isinstance(value, tuple):
                    f.write(f"{key} (tuple):\n")  # Scrive il tipo per chiarezza
                    # Converte la tupla in stringa per la scrittura
                    value = str(value)  # Trasforma la tupla in stringa
                else:
                    f.write(f"{key}:\n")
                
                # Scrive il valore nel file
                f.write(f"{value}\n")
                
    else:
        for i, (df_node, df_dpg_metrics, index, df_edge) in enumerate(zip(df_node_list, df_dpg_metrics_list, index_list, df_edge_list)):
            
            df_node.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_anomaly{index}_{args.mode_graph}_{args.mode_score}_node_metrics.csv'),
                        encoding='utf-8')    

            df_edge.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_anomaly{index}_{args.mode_graph}_{args.mode_score}_edge_metrics.csv'),
                        encoding='utf-8')
            with open(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_anomaly{index}_{args.mode_graph}_{args.mode_score}_dpg_metrics.txt'), 'w') as f:
                for key, value in df_dpg_metrics.items():
                    f.write(f"{key}:\n{value}\n")
