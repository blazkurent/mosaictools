from uncertain_variables import VariableSet
from .SubdomainClassifier import SubdomainClassifier
from .utils import find_typical_eigenvectors, flip_eigenvectors, calculate_MAC_matrix
import numpy as np
from gPCE_model import GpcModel

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
            
    def __init__(self, Q: VariableSet, max_freq_degree: int=4, max_vect_degree: int=5, resolution: float=0.6, classification_method: str='svc', **class_kwargs: dict):
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
        
        reference_vectors = find_typical_eigenvectors(eigenvectors, frequencies, resolution=self.resolution)
        labels = np.argmax(calculate_MAC_matrix(reference_vectors, eigenvectors), axis = 0)
        self.classifier.fit(params, labels)

        eigenvectors = flip_eigenvectors(eigenvectors, reference_vectors)
        
        self.reference_vectors = reference_vectors

        classes = self.classifier.get_classes()
        _, self.n_nodes = eigenvectors.shape

        self.eigenvector_models = {}
        self.frequency_models = {}

        for i in range(len(classes)):
            data_indexes = np.where(labels == classes[i])[0]

            num_data = len(data_indexes)

            vector_model = GpcModel(self.Q, 1)
            frequency_model = GpcModel(self.Q, 1)
            for j in range(2, self.max_freq_degree):
                if GpcModel(self.Q, j).basis.I.shape[0] > num_data:
                    break
                frequency_model = GpcModel(self.Q, j)

            for j in range(2, self.max_vect_degree):
                if GpcModel(self.Q, j).basis.I.shape[0] > num_data:
                    break
                vector_model = GpcModel(self.Q, j)
            
            vector_cluster = eigenvectors[data_indexes, :]

            vector_model.compute_coeffs_by_regression(params[data_indexes], vector_cluster)
            self.eigenvector_models[classes[i]] = vector_model

            frequency_cluster = frequencies[data_indexes]
            
            frequency_model.compute_coeffs_by_regression(params[data_indexes], frequency_cluster)
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
        eigenvectors = np.zeros((len(q), self.n_nodes))
        frequencies = np.zeros(len(q))
        u = np.unique(model_idx)
        for i in range(len(u)):
            indexes = np.where(model_idx == u[i])[0]
            q_i = q[indexes]
            frequencies[indexes] = self.frequency_models[u[i]].predict(q_i)
            eigenvectors[indexes, :] = self.eigenvector_models[u[i]].predict(q_i)
        return frequencies, eigenvectors

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
    
    def get_classification_results(self, q):
        ''' Perform classification on parameter samples `q`.

            Parameters
            ----------
            q : ndarray of shape (n_samples, n_parameters)
                Input parameters.
            Returns
            -------
            labels : ndarray of shape (n_samples, )
                Results of the classification.'''
        assert len(self.eigenvector_models) != 0, 'The ModeModel is not trained yet'
        model_idx = self.classifier.predict(q)
        return model_idx
    
    def get_reference_vectors(self) -> np.ndarray:
        return self.reference_vectors

    def get_labels(self) -> list:
        ''' Return the list of class labels of the trained classifier.
            
            Returns
            -------
            classes : list
                List of class labels of the trained classifier.'''
        
        classes = self.classifier.get_classes()
        return classes
