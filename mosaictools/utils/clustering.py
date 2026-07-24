import numpy as np
from typing import Union
from sklearn.cluster import AgglomerativeClustering

#----------- Clustering -------------#

def calculate_MAC_matrix(eigenvectors_1 : np.ndarray, eigenvectors_2: np.ndarray) -> np.ndarray:
    ''' Returns the MAC matrix between two arrays of eigenvectors.
    
        Parameters
        ----------
        eigenvectors_1 : ndarray
            Array of eigenvectors.

        eigenvectors_2 : ndarray
            Array of eigenvectors.
            
        Returns
        -------
        MAC : ndarray
            Matrix of MAC values.'''
    
    dim_1 = eigenvectors_1.shape[0]
    dim_2 = eigenvectors_2.shape[0]

    D = np.dot(eigenvectors_1,eigenvectors_2.T)
    D1 = np.dot(np.conjugate(eigenvectors_1),eigenvectors_1.T)
    D2 = np.dot(np.conjugate(eigenvectors_2),eigenvectors_2.T)
    A = np.tile(np.diag(D1)[:,np.newaxis],dim_2)
    B = np.tile(np.diag(D2)[:,np.newaxis],dim_1).T
    MAC = D**2/(A*B)

    return MAC

def _MAC_distance_matrix(eigenvectors : np.ndarray, method: str='inverse') -> np.ndarray:
    ''' Returns the matrix of MAC-distances calculated with method 'inverse' or 'difference'.
    
        Parameters
        ----------
        eigenvectors : ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
        
        method : {'inverse', 'difference'}, default='inverse'
            Distance calculation method.
            
        Returns
        -------
        dist_matrix : ndarray of shape (n_samples, n_samples)
            Matrix of MAC-distances'''
    
    dim = eigenvectors.shape[0]
    D = np.dot(eigenvectors,np.transpose(eigenvectors))
    A = np.tile(np.expand_dims(np.diag(D),axis=1),dim)
    B = np.transpose(A)
    if method=='inverse':
        dist_matrix = A*B/(D**2)-1
    elif method=='difference':
        dist_matrix = 1-D**2/(A*B)
    return dist_matrix

def _find_clusters(eigenvectors : np.ndarray, dist_matrix:Union[np.ndarray,None] = None, clustering_threshold: float=0.1) -> np.ndarray:
    ''' Returns the cluster centers using hierarchical clustering.
    
        Parameters
        ----------
        eigenvectors : ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
            
        dist_matrix : ndarray of shape (n_samples, n_samples), default=None
            Distance matrix used in hierarchical clustering, calculated in relation to the MAC values if missing.
            
        clustering_threshold : float, default=0.1
            Clustering threshold.
            
        Returns
        -------
        cluster_centers : ndarray pf shape (n_clusters, n_nodes)
            Array of the closest eigenvectors from each cluster centers.'''
        
    if dist_matrix is None:
        dist_matrix = _MAC_distance_matrix(eigenvectors, method='difference')
    clustering_model = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average',distance_threshold=clustering_threshold)
    clusters = clustering_model.fit_predict(dist_matrix)
    cluster_names, cluster_sizes = np.unique(clusters,return_counts=True)

    cluster_names_sorted = cluster_names[np.argsort(cluster_sizes)[::-1]]
    cluster_centers = []

    for cluster in cluster_names_sorted:
        ind_center = np.argmin(np.mean(dist_matrix[clusters==cluster][:,clusters==cluster],axis=0))
        cluster_center = eigenvectors[clusters==cluster][ind_center]
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)
    return cluster_centers

def _get_highest_off_diagonal_macs(cluster_centers : np.ndarray) -> np.ndarray:
    ''' Calculates the highest off-diagonal MAC values between the cluster centers.
    
        Parameters
        ----------
        cluster_centers : ndarray of shape (n_clusters, n_nodes)
            Array of cluster center eigenvectors.
            
        Returns
        -------
        highest_off_diagonal_macs : ndarray of shape (n_clusters, )
            Array of the highest off-diagonal MAC values.'''

    automac = calculate_MAC_matrix(cluster_centers,cluster_centers)
    num_of_clusters = cluster_centers.shape[0]
    highest_off_diagonal_macs = [0]
    for i in range(1,num_of_clusters):
        highest_mac = np.max(automac[i,:i])
        highest_off_diagonal_macs.append(highest_mac)
    highest_off_diagonal_macs = np.array(highest_off_diagonal_macs)
    return highest_off_diagonal_macs

def find_typical_eigenvectors(eigenvectors : np.ndarray, eigenfrequencies: np.ndarray, dist_matrix: Union[np.ndarray,None] = None, resolution: float=0.3, recursion_iteration: int=0, recursion_limit: int=10) -> np.ndarray:
    ''' Calculates the typical eigenvectors.
    
        Parameters
        ----------
        eigenvectors : ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
            
        eigenvfrequencies : ndarray of shape (n_samples, )
            Array of eigenvfrequencies.
            
        dist_matrix : ndarray of shape (n_samples, n_samples), default=None
            Distance matrix used in hierarchical clustering. Calculated in relation to the MAC values if missing.
        
        resolution : float, default=0.3
            Resolution of the subdomain segmentation.
            
        recursion_iteration : int=0
            Recursion iterator.

        recursion_limit : int=10
            Maximum number of iterations for the typical eigenvector calculation.
            
        Returns
        -------
        typical_eigenvectors : ndarray of shape (n_clusters, n_nodes)
            Array of typical eigenvectors.'''
    
    typical_eigenvectors = np.empty((0,eigenvectors.shape[1]))
    if recursion_iteration > recursion_limit:
        return   typical_eigenvectors

    if dist_matrix is None:
        dist_matrix = _MAC_distance_matrix(eigenvectors, method='difference')
    # try:
    cluster_centers = _find_clusters(eigenvectors, dist_matrix=dist_matrix, clustering_threshold=0.1)
    # except:
        # return   typical_eigenvectors

    highest_off_diagonal_macs = _get_highest_off_diagonal_macs(cluster_centers)
    ind_clusters_retain = highest_off_diagonal_macs<0.2
    typical_eigenvectors = np.vstack((typical_eigenvectors,cluster_centers[ind_clusters_retain]))
    mac = calculate_MAC_matrix(eigenvectors,typical_eigenvectors)
    ind_eignevectors_unclustered = np.max(mac,axis=1)<resolution
    if np.sum(ind_eignevectors_unclustered)>0:
        eigenvectors_unclustered = eigenvectors[ind_eignevectors_unclustered]
        
        if len(eigenvectors_unclustered) == 1:
            typical_eigenvectors_remaining = eigenvectors_unclustered
        else:
            dist_matrix_unclustered = dist_matrix[ind_eignevectors_unclustered][:,ind_eignevectors_unclustered]
            typical_eigenvectors_remaining = find_typical_eigenvectors(eigenvectors_unclustered,
                                                                    eigenfrequencies=None,
                                                                    dist_matrix=dist_matrix_unclustered,
                                                                    resolution=resolution,
                                                                    recursion_iteration=recursion_iteration+1,
                                                                    recursion_limit=recursion_limit)

        typical_eigenvectors = np.vstack((typical_eigenvectors,typical_eigenvectors_remaining))

    if recursion_iteration==0 and eigenfrequencies is not None:
        mac = calculate_MAC_matrix(typical_eigenvectors,eigenvectors)
        mean_eigenfrequencies = []
        for i in range(typical_eigenvectors.shape[0]):
            mask = np.where(np.argmax(mac,axis=0)==i)
            mean_eigenfrequencies.append(np.mean(eigenfrequencies[mask]))
        ind_sort = np.argsort(mean_eigenfrequencies)
        typical_eigenvectors = typical_eigenvectors[ind_sort]
    return typical_eigenvectors