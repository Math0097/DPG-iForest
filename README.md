# DPG-iForest

DPG-iForest is an extension of DPG, a model-agnostic tool aimed at improving the transparency and interpretability of tree-based ensemble models. While the classic DPG relies on three-element predicates (feature, operator, and value), DPG-iForest simplifies this structure, focusing solely on two-element predicates, consisting of a feature and an operator.

This streamlined approach reflects the nature of the Isolation Forest algorithm, emphasizing the relationship between features and their role in decisions, without the need for a value element. DPG-iForest utilizes a graph structure to capture and visualize the model's logic, enabling users to see how features interact with operators and influence predictions. This method sheds light on key decision points and provides deeper insights into the overall decision-making process.

Additionally, DPG-iForest introduces the Inlier-Outlier Propagation Score, a metric that enhances understanding of how the model determines whether data points are "inliers" or "outliers." By focusing on feature-operator relationships, this metric offers a clearer, more transparent view of the model's internal mechanisms and enables more accurate comparisons between different features.
<p align="center">
  <img src="https://github.com/Math0097/DPG-iForest/blob/main/examples/sintetic_4_200_preprocessed_iForest_bl200_dec2_feature_operator_all_freq_pes_temp.png" width="1000" />
</p>

## The structure
The concept behind DPG is to convert a generic tree-based ensemble model for classification into a graph, where:
- Nodes represent predicates, i.e., the feature-value associations present in each node of every tree;
- Edges denote the frequency with which these predicates are satisfied during the model training phase by the samples of the dataset.

<p align="center">
  <img src="https://github.com/Math0097/DPG-iForest/blob/main/examples/example.png?raw=true" width="600" />
</p>

## Metrics
The graph-based nature of DPG-iForest provides significant enhancements in the direction of a complete mapping of the ensemble structure.
| Property     | Definition | Utility |
|--------------|------------|---------|
| _Inlier-Outlier Propagation Score_  | It evaluates the tendency of each node in the graph (representing a feature and its operator) to drive the decision-making process toward classifying data points as either `Outliers` or `Inliers`. |

## The DPG-iForest library

#### Main script
- `dpg_custom.py`: with this script it is possible to apply DPG-iForest to your dataset.

#### Metrics and visualization
The library also contains two other essential scripts:
- `core.py` contains all the functions used to calculate and create the DPG-iForest and the metrics.
- `visualizer.py` contains the functions used to manage the visualization of DPG-iForest.

#### Output
The DPG-iForest application, through `dpg_custom.py`, produces several files:
- the visualization of DPG-iForest in a dedicated environment, which can be zoomed and saved;
- a `.txt` file containing iForest generated paths and data points classified as outliers;
- a `.csv` file containing the information about all the nodes of the DPG-iForest and their associated Inlier-Outlier Propagation Score metric;

## Easy usage
Usage: `python dpg_custom.py --ds "./datasets/sintetic_4_200_preprocessed.csv" --l 200  --t 2 --cont 0.02 --seed 42 --dir "./iForest_results_clf/" --plot --save_plot_dir "./iForest_results_clf/" --attribute "Inlier-Outlier Propagation Score" --class_flag --edge_attribute "Weighted frequency" --predicates "feature_operator" --mode "global" --mode_graph "all" --mode_score "freq_pes"`
Where:
- `ds` is the name of the standard classification `sklearn` dataset to be analyzed;
- `l` is the number of base learners for the Random Forest;
- `t` is the decimal precision of each feature;
- `cont` is the contamination parameter of the iForest model;
- `seed` is the random seed;
- `dir` is the path of the directory to save the files;
- `plot` is a store_true variable which can be added to plot the DPG-iForest;
- `save_plot_dir` is the path of the directory to save the plot image;
- `attribute` is the specific node metric which can be visualized on the DPG-iForest;
- `class_flag` is a store_true variable which can be added to highlight class nodes.
- `edge_attribute` is the specific edge metric which can be visualized on the DPG-iForest;

#### Example `dpg_custom.py`
Some examples can be appreciated in the `examples` folder: https://github.com/Math0097/DPG-iForest/blob/main/examples

In particular, the following DPG-iForest is obtained by transforming a Isolation Forest with 200 base learners, trained on Annthyroid dataset.
The used command is `python dpg_custom.py --ds "./datasets/Annthyroid_preprocessed.csv" --l 200  --t 2 --cont 0.0361 --seed 42 --dir "./iForest_results_clf/" --plot --save_plot_dir "./iForest_results_clf/" --attribute "Inlier-Outlier Propagation Score" --class_flag --edge_attribute "Weighted frequency" --predicates "feature_operator" --mode "global" --mode_graph "all" --mode_score "freq_pes"`.
<p align="center">
  <img src="https://github.com/Math0097/DPG-iForest/blob/main/examples/Annthyroid_preprocessed_iForest_bl200_dec2_feature_operator_all_freq_pes_Inlier-OutlierPropagationScore.png" width="800" />
</p>
