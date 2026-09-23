import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .clustering import calculate_MAC_matrix

def plot_space_division(model, mode, n_samples=10000):
    """ Return a seaborn PairGrid object showing the division of the variables space for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
                        
        mode : int
            The mode for which the parametric space division is to be visualized.

        n_samples : int
            The number of variable samples to draw from the model for visualization.
            
        Returns
        -------
        g : seaborn PairGrid object
            A seaborn PairGrid object showing the division of the parametric space for the given mode"""

    if model.n_demo_samples is None or model.n_demo_samples != n_samples:
        model.demo_variables = model.sample(n_samples)
        model.demo_labels = model.get_class_labels(model.demo_variables)
        model.n_demo_samples = n_samples
    
    # labels = model.get_class_labels(model.demo_variables)
    n_clusters = len(np.unique(model.demo_labels[:, mode-1]))

    x_data = pd.DataFrame(model.demo_variables, columns=model.Q.variable_names())
    x_data.insert(3, "label", model.demo_labels[:, mode-1], True)
    x_data["label"] = x_data["label"].astype(int)

    palette=sns.color_palette("Paired", n_clusters)
    g = sns.PairGrid(x_data, hue="label", palette=palette)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Variable space division in mode {}".format(mode))

    return g

def plot_reference_eigenvectors(model, mode):
    """ Return a matplotlib figure showing the reference eigenvectors for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
                        
        mode : int
            The mode for which the reference eigenvectors are to be visualized.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure showing the reference eigenvectors for the given mode"""
    
    reference_eigenvectors = model.get_reference_vectors()[mode-1]

    n_clusters = len(reference_eigenvectors)

    palette=sns.color_palette("Paired", n_clusters)

    colors = iter(palette)


    fig, ax = plt.subplots(n_clusters)
    if n_clusters == 1:
        fig.suptitle("Reference eigenvector of mode {}".format(mode))
    else:
        fig.suptitle("Reference eigenvectors of mode {}".format(mode))

    if n_clusters == 1:
        ax.plot(reference_eigenvectors[0, :], color=next(colors))
    else:
        for i in range(n_clusters):
            ax[i].plot(reference_eigenvectors[i, :], color=next(colors), label=i)
            if i != n_clusters-1:
                ax[i].set_xticklabels([])
                ax[i].set_xticks([])
        ax[i].set_xlabel("Nodes")
        fig.legend()
    return fig

def plot_reference_correlation_matrix(model, mode):
    """ Return a matplotlib figure showing the correlation matrix of the reference eigenvectors for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
                        
        mode : int
            The mode for which the correlation matrix of the reference eigenvectors is to be visualized.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure showing the correlation matrix of the reference eigenvectors for the given mode"""
    
    reference_eigenvectors = model.get_reference_vectors()[mode-1]

    n_clusters = len(reference_eigenvectors)

    
    mac_matrix = calculate_MAC_matrix(reference_eigenvectors, reference_eigenvectors)

    fig = plt.figure(figsize=(6,6))
    fig.suptitle("Correlations between the reference eigenvectors of mode {}".format(mode))
    im = plt.imshow(mac_matrix, interpolation='none', vmin=0, vmax=1, aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(0, n_clusters, 1))
    ax.set_yticks(np.arange(0, n_clusters, 1))

    ax.set_xlabel("Reference eigenvectors")
    ax.set_ylabel("Reference eigenvectors")

    ax.set_xticklabels(np.arange(0, n_clusters, 1))
    ax.set_yticklabels(np.arange(0, n_clusters, 1))

    ax.set_xticks(np.arange(-.5, n_clusters, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_clusters, 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(n_clusters):
        for j in range(n_clusters):
            text = ax.text(j, i, round(mac_matrix[i, j], 2),
                        ha="center", va="center", color="k")
    cax=fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0,0.03,ax.get_position().height])
    cbar = plt.colorbar(im, cax = cax)
    cbar.set_label("MAC value")
    
    return fig