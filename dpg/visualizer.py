import os
import numpy as np
from graphviz import Source, Digraph
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from .utils import highlight_class_node, change_node_color, delete_folder_contents, change_edge_color

dot = Digraph()
dot.attr('node', fontname='Computer Modern Roman')
dot.attr('edge', fontname='Computer Modern Roman')

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif=['Computer Modern Roman'])


def plot_dpg(plot_name, dot, df, df_edge, df_dpg, save_dir="examples/", attribute=None, variant=True, communities=False, class_flag=False, edge_attribute=None):
    """
    Plots a Decision Predicate Graph (DPG) with various customization options.

    Args:
        plot_name: The name of the plot.
        dot: A Graphviz Digraph object representing the DPG.
        df: A pandas DataFrame containing node metrics.
        df_edge: A pandas DataFrame containing edge metrics.
        df_dpg: A pandas DataFrame containing DPG metrics.
        save_dir: Directory to save the plot image. Default is "examples/".
        attribute: A specific node attribute to visualize. Default is None.
        communities: Boolean indicating whether to visualize communities. Default is False.
        class_flag: Boolean indicating whether to highlight class nodes. Default is False.
        edge_attribute: A specific edge attribute to visualize. Default is None.

    Returns:
        None
    """

    # --------------------------
    # Coloring dei nodi (e degli archi) in base agli argomenti
    # --------------------------
    if attribute is None and not communities:
        # Schema di colori di base
        for index, row in df.iterrows():
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(157, 195, 230))  # Light blue per i nodi di classe
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(222, 235, 247))  # Light grey per gli altri nodi

    elif attribute is not None and not variant and not communities:
        # Colorazione dei nodi in base a un attributo specifico (nodi)
        colormap = cm.Blues
        # Evidenzia i nodi di classe se class_flag è True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Giallo per nodi di classe
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Escludi i nodi di classe dal processamento
        max_score = df[attribute].max()
        norm = mcolors.Normalize(0, max_score)
        colors = colormap(norm(df[attribute]))
        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255),
                                                  int(colors[index][1]*255),
                                                  int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
        plot_name = plot_name + f"_{attribute}".replace(" ","")

    elif attribute is not None and variant and not communities:
        # Colorazione dei nodi in base a un attributo specifico con una colormap custom
        colors = [(0, "#b31529"), (0.5, "#f9f9f9"), (1, "#1065ab")]  # blu-rosso
        colormap = LinearSegmentedColormap.from_list("my_cmap", colors)
        norm = plt.Normalize(vmin=-1, vmax=1)
        if class_flag:
            for index, row in df.iterrows():
                if 'Class -1' in row['Label']:
                    change_node_color(dot, row['Node'], '#b31529')
                    dot.node(str(row['Node']), label="Outlier", fontcolor="White", fontsize='54')
                elif 'Class 1' in row['Label']:
                    change_node_color(dot, row['Node'], '#1065ab')
                    dot.node(str(row['Node']), label="Inlier", fontcolor="White", fontsize='54')
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)
        color_values = norm(df[attribute])
        colors = colormap(color_values)
        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255),
                                                  int(colors[index][1]*255),
                                                  int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
            dot.node(str(row['Node']), fontsize='54')
        plot_name = plot_name + f"_{attribute}".replace(" ","")

    elif communities and attribute is None:
        # Colorazione in base alle comunità
        colormap = cm.YlOrRd
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)
        label_to_community = {label: idx for idx, s in enumerate(df_dpg['Communities']) for label in s}
        df['Community'] = df['Label'].map(label_to_community)
        max_score = df['Community'].max()
        norm = mcolors.Normalize(0, max_score)
        colors = colormap(norm(df['Community']))
        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255),
                                                  int(colors[index][1]*255),
                                                  int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
        plot_name = plot_name + "_communities"
    else:
        raise AttributeError("The plot can show the basic plot, communities or a specific node-metric")
    
    # --------------------------
    # Colorazione degli archi
    # --------------------------
    if edge_attribute is not None:
        colormap_edge = cm.Greys  # Colormap per gli archi (scala di grigi)
        max_edge_value = df_edge[edge_attribute].max()
        min_edge_value = df_edge[edge_attribute].min()
        norm_edge = mcolors.Normalize(vmin=min_edge_value, vmax=max_edge_value)
        for index, row in df_edge.iterrows():
            edge_value = row[edge_attribute]
            color = colormap_edge(norm_edge(edge_value))
            color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]*255),
                                                     int(color[1]*255),
                                                     int(color[2]*255))
            penwidth = 1 + 2 * norm_edge(edge_value)
            change_edge_color(dot, row['Source_id'], row['Target_id'], new_color=color_hex, new_width=penwidth)

    # Evidenzia i nodi di classe (eventuali modifiche specifiche)
    highlight_class_node(dot)
    
    '''
    # Conversione delle etichette: se trova "Feature0", "Feature1", … le sostituisce con "$F_{0}$", "$F_{1}$", ecc.
    import re

    def to_subscript(num_str):
        mapping = {
            "0": "\u2080",
            "1": "\u2081",
            "2": "\u2082",
            "3": "\u2083",
            "4": "\u2084",
            "5": "\u2085",
            "6": "\u2086",
            "7": "\u2087",
            "8": "\u2088",
            "9": "\u2089"
        }
        return "".join(mapping.get(ch, ch) for ch in num_str)

    pattern = r'(label=")Feature(\d+)([^"]*)"'
    def replace_feature(match):
        # match.group(1): 'label="'
        # match.group(2): il numero, es. "5"
        # match.group(3): il resto dell'etichetta (ad esempio " >" o " <=")
        return f'{match.group(1)}F{to_subscript(match.group(2))}{match.group(3)}"'

    dot.body = [re.sub(pattern, replace_feature, line) for line in dot.body]
    '''
    
    # (Opzionale) Conversione in notazione scientifica per le etichette
    def to_sci_notation(match):
        num = float(match.group(1))
        return f'label="{num:.2e}"'
    pattern = r'label=([0-9]+\.?[0-9]*)'
    # for i in range(len(dot.body)):
    #     dot.body[i] = re.sub(pattern, to_sci_notation, dot.body[i])
    
    # --------------------------
    # Rendering del grafo e preparazione dell'immagine
    # --------------------------
    dot.render("temp/" + plot_name, format="pdf")
    graph = Source(dot.source, format="png")
    graph.render("temp/" + plot_name + "_temp", view=False)
    img = Image.open("temp/" + plot_name + "_temp.png")

    # --------------------------
    # Creazione della figura principale che mostra il grafo con le barre dei colori
    # --------------------------
    fig = plt.figure(figsize=(16, 8))
    
    # Asse per il grafo
    ax_img = fig.add_axes([0.05, 0.15, 0.85, 0.75])
    ax_img.imshow(img)
    
    # Abilita il rendering con LaTeX e specifica il font
    #ax_img.set_title(r"Global representation of the Isolation Forest as a Graph", fontsize=30, pad=30)
    ax_img.axis('off')
    
    # Se esiste un attributo per i nodi, aggiungiamo la barra orizzontale sotto il grafo
    if attribute is not None:
        ax_hbar = fig.add_axes([0.11, 0.05, 0.8, 0.03]) # [left, bottom, width, height]
        cbar_nodes = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax_hbar, orientation='horizontal')
        cbar_nodes.set_label(attribute, fontsize=26)
        cbar_nodes.ax.xaxis.set_label_coords(0.5, -1.6)
    
    # Se esiste un attributo per gli archi, aggiungiamo la barra verticale a destra del grafo
    if edge_attribute is not None:
        ax_vbar = fig.add_axes([0.92, 0.1, 0.015, 0.8])
        cbar_edges = plt.colorbar(cm.ScalarMappable(norm=norm_edge, cmap=cm.Greys), cax=ax_vbar, orientation='vertical')
        cbar_edges.set_ticks([norm_edge.vmin, norm_edge.vmax])
        cbar_edges.set_label(edge_attribute, fontsize=26)
        cbar_edges.ax.yaxis.set_label_coords(1.7, 0.5)
    
    # Salvataggio della figura principale (grafo + barre) in PDF
    os.makedirs(save_dir, exist_ok=True)
    graph_pdf_path = os.path.join("temp", plot_name + "_graph.pdf")
    fig.savefig(graph_pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    
    
    # --------------------------
    # Salvataggio separato delle barre dei colori (se esistono)
    # --------------------------
    if attribute is not None:
        fig_nodes = plt.figure(figsize=(16, 8))
        ax_nodes = fig_nodes.add_axes([0.11, 0.05, 0.8, 0.03])
        cbar_nodes_pdf = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax_nodes, orientation='horizontal')
        cbar_nodes_pdf.set_label(attribute, fontsize=26)
        cbar_nodes_pdf.ax.xaxis.set_label_coords(0.5, -1.6)
        node_bar_pdf_path = os.path.join("temp", plot_name + "_colorbar_nodes.pdf")
        fig_nodes.savefig(node_bar_pdf_path, format="pdf", dpi=300, bbox_inches='tight')
        plt.close(fig_nodes)
    
    if edge_attribute is not None:
        fig_edges = plt.figure(figsize=(16, 8))
        ax_edges = fig_edges.add_axes([0.92, 0.1, 0.015, 0.8])
        cbar_edges_pdf = plt.colorbar(cm.ScalarMappable(norm=norm_edge, cmap=cm.Greys), cax=ax_edges, orientation='vertical')
        cbar_edges_pdf.set_ticks([norm_edge.vmin, norm_edge.vmax])  # Imposta solo i valori estremi
        cbar_edges_pdf.set_label(edge_attribute, fontsize=26)
        cbar_edges_pdf.ax.yaxis.set_label_coords(1.7, 0.5)
        edge_bar_pdf_path = os.path.join("temp", plot_name + "_colorbar_edges.pdf")
        fig_edges.savefig(edge_bar_pdf_path, format="pdf", dpi=300, bbox_inches='tight')
        plt.close(fig_edges)
        
    dot.render("temp/" + plot_name + "_render", format="pdf")    
    
    # Salvataggio della figura principale in PNG
    main_png_path = os.path.join(save_dir, plot_name + ".png")
    fig.savefig(main_png_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)

    # (Opzionale) Pulizia dei file temporanei
    # delete_folder_contents("temp")
