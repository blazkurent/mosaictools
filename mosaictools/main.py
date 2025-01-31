''' This module provides surrogate modelling tools using the Mode-Shape-Adapted Input Parameter Domain Cutting (MOSAIC) approach.'''

import numpy as np
from .gpc_surrogate import GpcSurrogateModel
from .simparameter_set import SimParamSet
from .simparameter import SimParameter
from .distributions import UniformDistribution
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from scipy.stats import qmc
import copy

#----------- Clustering -------------#

def _calculate_MAC_matrix(eigenvectors_1 : np.ndarray, eigenvectors_2: np.ndarray) -> np.ndarray:
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

def _calculate_diag_MAC_matrix(eigenvectors_1 : np.ndarray, eigenvectors_2 : np.ndarray) -> np.ndarray:
    ''' Compute the diagonal terms of the MAC matrix.
        Parameters
        ----------
        eigenvectors_1 : np.ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
        
        eigenvectors_2 : np.ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.

        Returns
        -------
        MAC_diag : np.ndarray of shape (n_samples, n_samples)
            Diagonal terms of the MAC matrix.'''
    
    assert eigenvectors_1.shape == eigenvectors_2.shape, "Eigenvectors must be of the same size"
    D = np.sum(eigenvectors_1 * np.conjugate(eigenvectors_2), axis=1)
    D1 = np.sum(eigenvectors_1 * np.conjugate(eigenvectors_1), axis=1)
    D2 = np.sum(eigenvectors_2 * np.conjugate(eigenvectors_2), axis=1)
    MAC_diag = (D**2)/(D1*D2)
    return MAC_diag

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

def _find_clusters(eigenvectors : np.ndarray, dist_matrix:np.ndarray | None = None, clustering_threshold: float=0.1) -> np.ndarray:
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

    automac = _calculate_MAC_matrix(cluster_centers,cluster_centers)
    num_of_clusters = cluster_centers.shape[0]
    highest_off_diagonal_macs = [0]
    for i in range(1,num_of_clusters):
        highest_mac = np.max(automac[i,:i])
        highest_off_diagonal_macs.append(highest_mac)
    highest_off_diagonal_macs = np.array(highest_off_diagonal_macs)
    return highest_off_diagonal_macs

def _find_typical_eigenvectors(eigenvectors : np.ndarray, eigenfrequencies: np.ndarray, dist_matrix: np.ndarray | None = None, resolution: float=0.3, recursion_iteration: int=0, recursion_limit: int=10) -> np.ndarray:
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
    mac = _calculate_MAC_matrix(eigenvectors,typical_eigenvectors)
    ind_eignevectors_unclustered = np.max(mac,axis=1)<resolution
    if np.sum(ind_eignevectors_unclustered)>0:
        eigenvectors_unclustered = eigenvectors[ind_eignevectors_unclustered]
        
        if len(eigenvectors_unclustered) == 1:
            typical_eigenvectors_remaining = eigenvectors_unclustered
        else:
            dist_matrix_unclustered = dist_matrix[ind_eignevectors_unclustered][:,ind_eignevectors_unclustered]
            typical_eigenvectors_remaining = _find_typical_eigenvectors(eigenvectors_unclustered,
                                                                    eigenfrequencies=None,
                                                                    dist_matrix=dist_matrix_unclustered,
                                                                    resolution=resolution,
                                                                    recursion_iteration=recursion_iteration+1,
                                                                    recursion_limit=recursion_limit)

        typical_eigenvectors = np.vstack((typical_eigenvectors,typical_eigenvectors_remaining))

    if recursion_iteration==0 and eigenfrequencies is not None:
        mac = _calculate_MAC_matrix(typical_eigenvectors,eigenvectors)
        mean_eigenfrequencies = []
        for i in range(typical_eigenvectors.shape[0]):
            mask = np.where(np.argmax(mac,axis=0)==i)
            mean_eigenfrequencies.append(np.mean(eigenfrequencies[mask]))
        ind_sort = np.argsort(mean_eigenfrequencies)
        typical_eigenvectors = typical_eigenvectors[ind_sort]
    return typical_eigenvectors

#------------ Flipping and normalization --------------#

def _normalize_eigenvectors(input_eigenvectors: np.ndarray) -> np.ndarray:
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

def _flip_eigenvectors(input_eigenvectors: np.ndarray, reference_eigenvectors: np.ndarray | None = None) -> np.ndarray:
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

#----------- Classification ---------------#

class SubdomainClassifier():
    ''' Classifier used in a ModeModel class.
            
        Attributes
        ----------
        classes: list
            List of class labels.

        model : object
            Classifier model set with parameters `method` and `**kwargs`.
            
        Methods
        -------
        __init__(self, method='svc', classifier=None, **kwargs)
            SubdomainClassifier constructor.

        fit(self, X, y)
            Fit the classifier model according to the given training data.
                
        predict(self, X)
            Perform classification on samples in `X`.
                
        predict_probability(self, X)
            Compute probabilities of possible outcomes for samples in `X`.
            
        score(self, X, y)
            Return the mean accuracy on the given test data and labels.
        
        get_classes(self)
            Return the list of class labels of the trained classifier.'''
                
    def __init__(self, method: str='svc', **kwargs: dict):
        ''' SubdomainClassifier constructor.

            Parameters
            ----------
            method : {'svc', 'custom'}, default='svc'
                The classification method of the SubdomainClassifier.

            **kwargs : dict
                Arbitrary keyword arguments of the classifier model. It should be a classifier instance in the format `classifier_model=model` while using method='custom', or the parameters of the SVC model with method='svc'.'''
        
        self.classes = []
        self.model = None
        if method == 'svc':
            if 'classifier_model' in kwargs:
                del kwargs['classifier_model']
            self.model = SVC(**kwargs)
        elif method == 'custom':
            assert 'classifier_model' in kwargs, "A classifier model instance is requred as parameter `classifier_model` while using method='custom'."
            self.model = copy.deepcopy(kwargs['classifier_model'])

        else:
            raise ValueError("method='{}' is not supported".format(method))

    def fit(self, X: np.ndarray, y: np.ndarray):
        ''' Fit the classifier model according to the given training data.
        
            Parameters
            ----------
            X : ndarray of shape (n_samples, n_parameters)
                Training parameters.

            y : ndarray of shape (n_samples,)
                Sample labels.'''
        
        self.classes = np.unique(y)
        if len(self.classes) > 1:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        ''' Perform classification on samples in `X`.
        
            Parameters
            ----------
            X : ndarray of shape (n_samples, n_parameters)
                Input parameters.
                
            Returns
            -------
            y_pred : ndarray of shape (n_samples, )
                Array of predicted class labels.'''
        
        assert len(self.classes) != 0, 'The classifier is not trained yet.'
        if len(self.classes) == 1:
            y_pred = np.ones((len(X), ))
        else:
            y_pred = self.model.predict(X)
            return y_pred

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        ''' Compute probabilities of possible outcomes for samples in `X`.

            Parameters
            ----------
            X : ndarray of shape (n_samples, n_parameters)
                Input parameters.
            
            Returns
            -------
            probabilities : ndarray of shape (n_samples, n_classes)
                Array of the prediction probabilities.'''
        
        assert len(self.classes) != 0, 'The classifier is not trained yet'
        if len(self.classes) == 1:
            probabilities = np.ones((len(X), 1))
        else:
            probabilities = self.model.predict_proba(X)
            return probabilities

            
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        ''' Return the mean accuracy on the given test data and labels.
        
            Parameters
            ----------
            X : ndarray of shape (n_samples, n_parameters)
                Test samples.

            y : ndarray of shape (n_samples,)
                True labels for `X`.
                
            Returns
            -------
            score : float
                Mean accuracy of self.predict(X) with reference to `y`.'''
        
        assert len(self.classes) != 0, 'The classifier is not trained yet.'
        if len(self.classes) == 1:
            score = 1.0
        else:
            score = self.model.score(X, y)   
        return score      
        
    def get_classes(self) -> list:
        ''' Return the list of class labels of the trained classifier.
            
            Returns
            -------
            classes : list
                List of class labels of the trained classifier.'''
        
        assert len(self.classes) != 0, 'The classifier is not trained yet'
        classes = self.classes.astype(int)
        return classes
    
#----------------- Surrogate models ----------------#

class ModeModel():
    ''' The surrogate model of one mode.
        
        Attributes
        ----------
        Q : SimParamSet
            Description of the parameters.

        max_freq_degree : int
            The maximum possible degree of approximation used in training the GPC models of the eigenfrequencies.
        
        max_vect_degree : int
            The maximum possible degree of approximation used in training the GPC models of the eigenvectors.
        
        resolution : float
            Resolution of the subdomain segmentation in the parameter space.
        
        classifier : SubdomainClassifier
            Sundomain classifier of the mode.
        
        n_nodes : int
            Number of nodes of the eigenvectors.
        
        eigenvector_models : dict
            Dictionary of the eigenvector surrogate models of the subdomains.
        
        frequency_models : dict
            Dictionary of the frequency surrogate models of the subdomains.
            
        Methods
        -------
        __init__(self, Q, max_freq_degree=4, max_vect_degree=5, resolution=0.6, classification_method='svc', **class_kwargs)
            ModeModel constructor.
                        
        fit(self, params, frequencies, eigenvectors)
            Fit the ModeModel according to the given training data.
                        
        predict(self, q)
            Perform prediction of the frequencies and eigenvectors on parameter samples `q`.
                
        predict_probability(self, q)
            Compute probabilities of possible outcomes for samples in `q`.
            
        get_classifier_accuracy(self, test_params, test_labels)
            Return the mean accuracy of the classifier on the given test data and labels.
        
        get_labels(self)        
            Return the list of class labels of the trained classifier.'''
            
    def __init__(self, Q: SimParamSet, max_freq_degree: int=4, max_vect_degree: int=5, resolution: float=0.6, classification_method: str='svc', **class_kwargs: dict):
        ''' ModeModel constructor.
        
            Parameters
            ----------
            Q : SimParamSet
                Description of the parameters.

            max_freq_degree : int, default=4
                The maximum possible degree of approximation used in training the GPC models of the eigenfrequencies.
            
            max_vect_degree : int, default=5
                The maximum possible degree of approximation used in training the GPC models of the eigenvectors.
            
            resolution : float, default=0.6
                Resolution of the subdomain segmentation in the parameter space.
            
            classification_method : {'svc', 'custom'}, default='svc'
                The classification method of the classifier model.
                        
            **class_kwargs : dict
                Arbitrary keyword arguments of the classifier model.'''
        
        self.Q = Q
        self.max_freq_degree=max_freq_degree
        self.max_vect_degree=max_vect_degree
        self.resolution = resolution
        self.classifier = SubdomainClassifier(method=classification_method, **class_kwargs)

        self.n_nodes = None
        self.eigenvector_models = {}
        self.frequency_models = {}

    def fit(self, params: np.ndarray, frequencies: np.ndarray, eigenvectors: np.ndarray):
        ''' Fit the ModeModel according to the given training data.
        
            Parameters
            ----------
            params : ndarray of shape (n_samples, n_parameters)
                Training parameters.
            
            frequencies : ndarray of shape (n_samples, )
                Training frequencies.
            
            eigenvectors : ndarray of shape (n_samples, n_nodes)
                Training eigenvectors.'''
        
        reference_vectors = _find_typical_eigenvectors(eigenvectors, frequencies, resolution=self.resolution)
        labels = np.argmax(_calculate_MAC_matrix(reference_vectors, eigenvectors), axis = 0)
        self.classifier.fit(params, labels)

        eigenvectors = _flip_eigenvectors(eigenvectors, reference_vectors)

        classes = self.classifier.get_classes()
        _, self.n_nodes = eigenvectors.shape

        self.eigenvector_models = {}
        self.frequency_models = {}

        for i in range(len(classes)):
            data_indexes = np.where(labels == classes[i])[0]

            num_data = len(data_indexes)

            vector_model = GpcSurrogateModel(self.Q, 1)
            frequency_model = GpcSurrogateModel(self.Q, 1)
            for j in range(2, self.max_freq_degree):
                if GpcSurrogateModel(self.Q, j).basis.I.shape[0] > num_data:
                    break
                frequency_model = GpcSurrogateModel(self.Q, j)

            for j in range(2, self.max_vect_degree):
                if GpcSurrogateModel(self.Q, j).basis.I.shape[0] > num_data:
                    break
                vector_model = GpcSurrogateModel(self.Q, j)
            
            vector_cluster = eigenvectors[data_indexes, :]

            vector_model.compute_coeffs_by_regression(params[data_indexes].T, vector_cluster.T)
            self.eigenvector_models[classes[i]] = vector_model

            frequency_cluster = frequencies[data_indexes]
            
            frequency_model.compute_coeffs_by_regression(params[data_indexes].T, frequency_cluster.T)
            self.frequency_models[classes[i]] = frequency_model

    def predict(self, q : np.ndarray) -> tuple:
        ''' Perform prediction of the frequencies and eigenvectors on parameter samples `q`.

            Parameters
            ----------
            q : ndarray of shape (n_samples, n_parameters)
                Input parameters.

            Returns
            -------
            frequencies : ndarray of shape (n_samples, )
                Predicted frequencies.

            eigenvectors : ndarray of shape (n_samples, n_nodes)
                Predicted eigenvectors.'''
        
        assert len(self.eigenvector_models) != 0, 'The ModeModel is not trained yet'
        model_idx = self.classifier.predict(q)
        eigenvectors = np.zeros((self.n_nodes, len(q)))
        frequencies = np.zeros(len(q))
        u = np.unique(model_idx)
        for i in range(len(u)):
            indexes = np.where(model_idx == u[i])[0]
            q_i = q[indexes]
            frequencies[indexes] = self.frequency_models[u[i]].predict_response(q_i.T)
            eigenvectors[:, indexes] = self.eigenvector_models[u[i]].predict_response(q_i.T)
        return frequencies, eigenvectors.T

    def predict_probability(self, q : np.ndarray) -> np.ndarray:
        ''' Compute probabilities of possible outcomes for samples in `q`.

            Parameters
            ----------
            q : ndarray of shape (n_samples, n_parameters)
                Input parameters.
                
            Returns
            -------
            probabilities : ndarray of shape (n_samples, n_classes)
                Array of the prediction probabilities.'''
        
        assert len(self.eigenvector_models) != 0, 'The ModeModel is not trained yet'
        probabilities = self.classifier.predict_probability(q)
        return probabilities
    
    def get_classifier_accuracy(self, test_params : np.ndarray, test_labels : np.ndarray) -> float:
        ''' Return the mean accuracy of the classifier on the given test data and labels.
        
            Parameters
            ----------
            test_params : ndarray of shape (n_samples, n_parameters)
                Test samples.

            test_labels : ndarray of shape (n_samples,)
                True labels for `test_params`.
                
            Returns
            -------
            score : float
                Mean accuracy of the SubdomainClassifier model with reference to `test_labels`.'''
        
        accuracy = self.classifier.score(test_params, test_labels)
        return accuracy
    
    def get_labels(self) -> list:
        ''' Return the list of class labels of the trained classifier.
            
            Returns
            -------
            classes : list
                List of class labels of the trained classifier.'''
        
        classes = self.classifier.get_classes()
        return classes
        
class Mosaic():
    ''' The global surrogate model.

        Attributes
        ----------
        Q : SimParamSet
            Description of the parameters.

        mode_models : dict
            Dictionaty of ModeModels.

        n_modes : int
            Number of modes.

        n_nodes : int
            Number of nodes in the eigenvectors.

        resolution : float
            Resolution of the subdomain segmentation in the parameter space.

        max_freq_degree : int
            The maximum possible degree of approximation used in training the GPC models of the eigenfrequencies.
        
        max_vect_degree : int
            The maximum possible degree of approximation used in training the GPC models of the eigenvectors.

        classification_method : {'svc', 'custom'}, default='svc'
            The classification method used in the ModeModels.
                
        class_kwargs : dict
            Arbitrary keyword arguments of the classifier model used in the ModeModels.

        Methods
        -------
        __init__(self, names, bounds, resolution=0.6, max_freq_degree=4, max_vect_degree=5, classification_method='svc', classifier=None, **kwargs_class)
            Mosaic constructor.
        
        fit(self, params, frequencies, eigenvectors, verbose=True)
            Fit the Mosaic according to the given training data.

        predict(self, q, reorder=False):
            Perform prediction of the frequencies and eigenvectors on parameter samples `q`.

        predict_probability(self, q):
            Compute probabilities of possible outcomes for samples in `q`.

        get_number_of_classes(self):
            Return array of class numbers of the trained classifiers in each mode.

        sample(self, n_samples):
            Return array of parameter samples using Halton.

        calculate_relative_frequency_errors(self, real_frequencies, predicted_frequencies):
            Return array of relative frequency errors between real frequency values and predictions.

        calculate_eigenvector_MAC_errors(self, real_eigenvectors, predicted_eigenvectors):
            Return array of 1 - MAC values between real eigenvectors and predictions.'''
    
    def __init__(self, names: list, bounds: list, resolution: float=0.6, max_freq_degree: int=4, max_vect_degree: int=5, classification_method: str='svc', **class_kwargs: dict):
        ''' Mosaic constructor.
            
            Parameters
            ----------
            names : list
                List of the parameter names.

            bounds : list
                List of tuples of upper and lower bounds of the parameters.
            
            resolution : float, default=0.6
                Resolution of the subdomain segmentation in the parameter space.

            max_freq_degree : int, default=4
                The maximum possible degree of approximation used in training the GPC models of the eigenfrequencies.
            
            max_vect_degree : int, default=5
                The maximum possible degree of approximation used in training the GPC models of the eigenvectors.
                
            classification_method : {'svc', 'custom'}, default='svc'
                The classification method of the SubdomainClassifiers.
                                
            **class_kwargs : dict
                Arbitrary keyword arguments of the classifier model.'''
        
        assert classification_method in ['svc', 'custom'], "Method `{}` is not supported for classification. Use `svc` or `custom`.".format(classification_method)
        if classification_method == 'custom':
            assert 'classifier_model' in class_kwargs, "Add a chosen classifier instance for method `custom`."
        assert len(names)==len(bounds), "The number of names and bounds should be equal."
        self.Q = SimParamSet()
        for i in range(len(names)):
            assert len(bounds[i]) == 2, "All bounds should be a tuple with two values."
            self.Q.add(SimParameter(names[i], UniformDistribution(bounds[i][0], bounds[i][1])))

        self.mode_models = {}
        self.resolution=resolution
        self.max_freq_degree=max_freq_degree
        self.max_vect_degree=max_vect_degree
        self.classification_method = classification_method
        
        self.n_modes = None
        self.n_nodes = None

        self.class_kwargs = class_kwargs

    def fit(self, params: np.ndarray, frequencies: np.ndarray, eigenvectors: np.ndarray, verbose: bool=True):
        ''' Fit the Mosaic according to the given training data.
        
            Parameters
            ----------
            params : ndarray of shape (n_samples, n_parameters)
                Training parameters.
            
            frequencies : ndarray of shape (n_samples, )
                Training frequencies.
            
            eigenvectors : ndarray of shape (n_samples, n_nodes)
                Training eigenvectors.

            verbose : boolean, default=True
                If true, the function prints the training progress.'''
        
        assert len(params.shape) == 2, "The dimensions of train parameter values should be [n_datapoints, n_params]."
        assert params.shape[1] == self.Q.num_params(), "The model constructed with {} parameters, but {} was given for the training. Change the model or the training settings.".format(self.Q.num_params(), params.shape[1])

        assert len(frequencies.shape) == 2, "The dimensions of training eigenfrequencies should be [n_datapoints, n_modes]."
        assert len(eigenvectors.shape) == 3, "The dimensions of training eigenvectors should be [n_datapoint, n_modes, n_nodes]."

        assert params.shape[0] == frequencies.shape[0] == eigenvectors.shape[0], "The training parameters, frequencies and eigenvectors should have the same number of datapoints."
        assert frequencies.shape[1] == eigenvectors.shape[1], "The training frequencies and eigenvectors should have the same number of modes."

        self.mode_models = {}

        _, n_modes, n_nodes = eigenvectors.shape
        self.n_modes = n_modes
        self.n_nodes = n_nodes

        params = self.Q.params2germ(params.T).T

        eigenvectors = _normalize_eigenvectors(eigenvectors)

        if verbose:
            print('Number of modes: {}'.format(n_modes))

        for i in range(n_modes):
            mode_model = ModeModel(self.Q, self.max_freq_degree, self.max_vect_degree, self.resolution, self.classification_method, **self.class_kwargs)
            mode_model.fit(params, frequencies[:, i], eigenvectors[:, i, :])
            self.mode_models[i] = mode_model
            if verbose:
                print('Mode {} is trained'.format(i+1))

    def predict(self, q : np.ndarray, reorder : bool=False) -> tuple:
        ''' Perform prediction of the frequencies and eigenvectors on parameter samples `q`.

            Parameters
            ----------
            q : ndarray of shape (n_samples, n_parameters)
                Input parameters.
            
            reorder : boolean, default=False
                If True, the function orders the values of the modes by the predicted frequencies.

            Returns
            -------
            frequencies : ndarray of shape (n_samples, n_modes)
                Predicted frequencies.

            eigenvectors : ndarray of shape (n_samples, n_modes, n_nodes)
                Predicted eigenvectors.'''
        
        assert len(self.mode_models) != 0, 'The MOSAIC model is not trained yet'
        
        q = self.Q.params2germ(q.T).T

        frequencies = np.zeros((len(q), self.n_modes))
        eigenvectors = np.zeros((len(q), self.n_modes, self.n_nodes))
        for i in range(self.n_modes):
            frequencies[:, i], eigenvectors[:, i, :] = self.mode_models[i].predict(q)
        if reorder == True:
            order = np.argsort(frequencies, axis=1)
            for j in range(len(q)):
                frequencies[j, :] = frequencies[j, :][order[j, :]]
                eigenvectors[j, :, :] = eigenvectors[j, :, :][order[j, :]]
        return frequencies, eigenvectors

    def predict_probability(self, q: np.ndarray) -> list:
        ''' Compute probabilities of possible outcomes for samples in `q`.

            Parameters
            ----------
            q : ndarray of shape (n_samples, n_parameters)
                Input parameters.
                
            Returns
            -------
            probabilities : list of ndarrays with length n_modes.
                List of the prediction probabilities.'''
        
        assert len(self.mode_models) != 0, 'The MOSAIC model is not trained yet.'
        q = self.Q.params2germ(q.T).T
        probabilities = []
        for i in range(self.n_modes):
            mode_probabilities = self.mode_models[i].predict_probability(q)
            probabilities.append(mode_probabilities)
        return probabilities

    def get_number_of_classes(self) -> list:
        ''' Return array of class numbers of the trained classifiers in each mode.
            
            Returns
            -------
            classes : ndarray of shape (n_modes, )
                List of class labels of the trained classifier.'''
        
        assert len(self.mode_models) != 0, 'The MOSAIC model is not trained yet.'
        classes = np.zeros(self.n_modes)
        for i in range(self.n_modes):
            classes[i] = len(self.mode_models[i].get_labels())
        return classes
    
    def sample(self, n_samples : int) -> np.ndarray:
        ''' Return array of sampled parameters using Halton.

            Parameters
            ----------
            n_samples : int
                Number of sample points.
            
            Returns
            -------
            parameters : ndarray of shape (n_samples, n_parameters)
                Array of sampled parameter values.'''
        
        n_params = self.Q.num_params()
        sampler = qmc.Halton(d=n_params, scramble=True)
        sample = sampler.random(n=n_samples)
        l_bounds = [-1] * n_params
        u_bounds = [1] * n_params
        parameters = qmc.scale(sample, l_bounds, u_bounds)
        parameters = self.Q.germ2params(parameters.T).T
        return parameters
    
    def calculate_relative_frequency_errors(self, real_frequencies : np.ndarray, predicted_frequencies : np.ndarray) -> np.ndarray:
        ''' Return array of relative frequency errors between real frequency values and predictions.

            Parameters
            ----------
            real_frequencies : ndarray of shape (n_samples, n_modes)
                Real frequency values.

            predicted_frequencies : ndarray of shape (n_samples, n_modes)
                Predicted frequency values.

            Returns
            -------
            frequency_errors : ndarray of shape (n_samples, n_modes)
                Relative frequency errors between real and predicted frequency values.'''
        
        assert real_frequencies.shape == predicted_frequencies.shape, 'The real and the predicted frequences should have the sem dimensions, but {} and {} was given.'.format(real_frequencies.shape, predicted_frequencies.shape)
        frequency_errors = np.abs(real_frequencies - predicted_frequencies)/real_frequencies
        return frequency_errors

    def calculate_eigenvector_MAC_errors(self, real_eigenvectors : np.ndarray, predicted_eigenvectors : np.ndarray) -> np.ndarray:
        ''' Return array of 1 - MAC values between real eigenvectors and predictions.

            Parameters
            ----------
            real_eigenvectors : ndarray of shape (n_samples, n_modes, n_nodes)
                Real eigenvectors.

            predicted_eigenvectors : ndarray of shape (n_samples, n_modes)
                Predicted eigenvectors.

            Returns
            -------
            mac_errors : ndarray of shape (n_samples, n_modes)
                1 - MAC values between real and predicted eigenvectors.'''
        
        mac_errors = np.zeros((real_eigenvectors.shape[0], self.n_modes))
        for i in range(self.n_modes):
            mac_errors[:, i] = 1 - _calculate_diag_MAC_matrix(predicted_eigenvectors[:, i, :], real_eigenvectors[:, i, :])
        return mac_errors
    
    # def save(self, path):
    #     TODO
    
    # def load(self, path):
    #     TODO
