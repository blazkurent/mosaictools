import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .clustering import calculate_MAC_matrix

def get_space_division(model, parameters, mode):
    """ Return a seaborn PairGrid object showing the division of the parametric space for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
            
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Array of parameters used for the classification.
            
        mode : int
            The mode for which the parametric space division is to be visualized.
            
        Returns
        -------
        g : seaborn PairGrid object
            A seaborn PairGrid object showing the division of the parametric space for the given mode"""
    
    labels = model.get_class_labels(parameters)
    n_clusters = len(np.unique(labels[:, mode-1]))

    x_data = pd.DataFrame(parameters, columns=model.Q.variable_names())
    x_data.insert(3, "label", labels[:, mode-1], True)
    x_data["label"] = x_data["label"].astype(int)

    palette=sns.color_palette("Paired", n_clusters)
    g = sns.PairGrid(x_data, hue="label", palette=palette)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Parametric space division in mode {}".format(mode))

    return g

def get_reference_eigenvectors(model, parameters, mode):
    """ Return a matplotlib figure showing the reference eigenvectors for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
            
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Array of parameters used for the classification.
            
        mode : int
            The mode for which the reference eigenvectors are to be visualized.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure showing the reference eigenvectors for the given mode"""
    
    labels = model.get_class_labels(parameters)
    n_clusters = len(np.unique(labels[:, mode-1]))

    palette=sns.color_palette("Paired", n_clusters)

    colors = iter(palette)

    reference_eigenvectors = model.get_reference_vectors()[mode-1]

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

def get_reference_correlation_matrix(model, parameters, mode):
    """ Return a matplotlib figure showing the correlation matrix of the reference eigenvectors for a given mode.
    
        Parameters
        ----------
        model : Mosaic
            A Mosaic model object.
            
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Array of parameters used for the classification.
            
        mode : int
            The mode for which the correlation matrix of the reference eigenvectors is to be visualized.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure showing the correlation matrix of the reference eigenvectors for the given mode"""
    
    labels = model.get_class_labels(parameters)
    n_clusters = len(np.unique(labels[:, mode-1]))

    reference_eigenvectors = model.get_reference_vectors()[mode-1]
    
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