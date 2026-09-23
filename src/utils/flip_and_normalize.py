import numpy as np
from typing import Union



#------------ Flipping and normalization --------------#

def normalize_eigenvectors(input_eigenvectors: np.ndarray) -> np.ndarray:
    ''' Normalizes the given eigenvectors.
    
        Parameters
        ----------
        input_eigenvectors : ndarray of shape (n_samples, n_nodes) or (n_samples, n_modes, n_nodes)
            Array of eigenvectos.
            
        Returns
        -------
        normalized_eigenvectors : ndarray with the shape of input_eigenvectors
            Array of normalized eigenvectors.'''
    
    if input_eigenvectors.ndim==2:
        norms = np.linalg.norm(input_eigenvectors, axis=1, keepdims=True)
    elif input_eigenvectors.ndim==3:
        norms = np.linalg.norm(input_eigenvectors, axis=2, keepdims=True)
    else:
        raise KeyError(f'The input has unexpected dimension of {input_eigenvectors.ndim}.')
    normalized_eigenvectors = input_eigenvectors / norms
    return normalized_eigenvectors

def _compute_dot_products(input_eigenvectors : np.ndarray, reference_eigenvectors: np.ndarray) -> np.ndarray:
    ''' Calculates dot products between eigenvectors and reference eigenvectors.
    
        Parameters
        ----------
        input_eigenvectors : ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
            
        reference_eigenvectors : ndarray of shape (n_clusters, n_nodes)
            Array of reference_eigenvectors.
            
        Returns
        -------
        dot_product : ndarray of shape (n_samples, 1, n_clusters)
            Matrix of dot products between input eigenvectors and typical eigenvectors.'''
    
    num_of_samples = input_eigenvectors.shape[0]
    num_of_modes = input_eigenvectors.shape[1]
    num_of_components = input_eigenvectors.shape[2]
    input_eigenvectors_reshaped = input_eigenvectors.reshape(-1, num_of_components)
    dot_products = np.dot(input_eigenvectors_reshaped, reference_eigenvectors.T)
    dot_products = dot_products.reshape(num_of_samples, num_of_modes, -1)
    return dot_products

def flip_eigenvectors(input_eigenvectors: np.ndarray, reference_eigenvectors: Union[np.ndarray,None] = None) -> np.ndarray:
    ''' Flips eigenvectors by clusters to the direction of the typical eigenvectors.
    
        Parameters
        ----------
        input_eigenvectors : ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
            
        reference_eigenvectors : ndarray of shape (n_clusters, n_nodes), default=None
            Array of reference_eigenvectors.
        
        Returns
        -------
        flipped_eigenvectors : ndarray
            Flipped versions of the input eigenvectors.'''
    
    input_eigenvectors = np.expand_dims(input_eigenvectors, axis=1)
    reference_eigenvectors = input_eigenvectors[0].copy() if reference_eigenvectors is None else reference_eigenvectors
    dot_products = _compute_dot_products(input_eigenvectors, reference_eigenvectors)
    max_abs_indices = np.argmax(np.abs(dot_products), axis=-1)
    max_abs_signs = np.sign(dot_products[np.arange(dot_products.shape[0])[:, None], np.arange(dot_products.shape[1]), max_abs_indices])
    flipped_eigenvectors = input_eigenvectors*max_abs_signs[:,:,np.newaxis]
    flipped_eigenvectors = flipped_eigenvectors.reshape(flipped_eigenvectors.shape[0], flipped_eigenvectors.shape[2])
    return flipped_eigenvectors
