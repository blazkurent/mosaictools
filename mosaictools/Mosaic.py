from uncertain_variables import VariableSet, Variable, UniformDistribution
import numpy as np
from .utils import normalize_eigenvectors
from .ModeModel import ModeModel
from scipy.stats import qmc
import pickle
import os
import pandas as pd


class Mosaic():
    ''' The global surrogate model.

        Attributes
        ----------
        Q : VariableSet
            Description of the variables.

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
    
    def __init__(self, variable_set: VariableSet, resolution: float=0.6, max_freq_degree: int=4, max_vect_degree: int=5, classification_method: str='svc', **class_kwargs: dict):
        ''' Mosaic constructor.
            
            Parameters
            ----------
            variable_set : VariableSet
                Variable set of the model
            
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
        # assert len(names)==len(bounds), "The number of names and bounds should be equal."
        self.Q = variable_set
        # for i in range(len(names)):
        #     assert len(bounds[i]) == 2, "All bounds should be a tuple with two values."
        #     self.Q.add(Variable(names[i], UniformDistribution(bounds[i][0], bounds[i][1])))

        self.mode_models = {}
        self.resolution=resolution
        self.max_freq_degree=max_freq_degree
        self.max_vect_degree=max_vect_degree
        self.classification_method = classification_method
        
        self.n_modes = None
        self.n_nodes = None

        self.class_kwargs = class_kwargs

    def fit(self, params: np.ndarray, frequencies: np.ndarray=None, eigenvectors: np.ndarray=None, QoI: pd.DataFrame=None, verbose: bool=True):
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
        assert params.shape[1] == self.Q.num_variables(), "The model constructed with {} parameters, but {} was given for the training. Change the model or the training settings.".format(self.Q.num_params(), params.shape[1])

        assert len(frequencies.shape) == 2, "The dimensions of training eigenfrequencies should be [n_datapoints, n_modes]."
        assert len(eigenvectors.shape) == 3, "The dimensions of training eigenvectors should be [n_datapoint, n_modes, n_nodes]."

        assert params.shape[0] == frequencies.shape[0] == eigenvectors.shape[0], "The training parameters, frequencies and eigenvectors should have the same number of datapoints."
        assert frequencies.shape[1] == eigenvectors.shape[1], "The training frequencies and eigenvectors should have the same number of modes."

        self.mode_models = {}

        _, n_modes, n_nodes = eigenvectors.shape
        self.n_modes = n_modes
        self.n_nodes = n_nodes

        params = self.Q.variable2germ(params)

        eigenvectors = normalize_eigenvectors(eigenvectors)

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
        
        q = self.Q.variable2germ(q)

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
        q = self.Q.variable2germ(q)
        probabilities = []
        for i in range(self.n_modes):
            mode_probabilities = self.mode_models[i].predict_probability(q)
            probabilities.append(mode_probabilities)
        return probabilities
    
    def get_class_labels(self, q: np.ndarray):
        ''' Perform classification on parameter samples `q`.

            Parameters
            ----------
            q : ndarray of shape (n_samples, n_parameters)
                Input parameters.
            Returns
            -------
            labels : ndarray of shape (n_samples, n_modes)
                Results of the classification.'''
        assert len(self.mode_models) != 0, 'The MOSAIC model is not trained yet'
        
        q = self.Q.variable2germ(q)

        labels = np.zeros((len(q), self.n_modes))
        for i in range(self.n_modes):
            labels[:, i] = self.mode_models[i].get_classification_results(q)

        return labels
    
    def get_reference_vectors(self) -> list:
        assert len(self.mode_models) != 0, 'The MOSAIC model is not trained yet.'
        reference_vectors = []
        for i in range(self.n_modes):
            reference_vectors.append(self.mode_models[i].get_reference_vectors())
        return reference_vectors


    def get_number_of_subdomains(self) -> list:
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
        
        n_params = self.Q.num_variables()
        sampler = qmc.Halton(d=n_params, scramble=True)
        sample = sampler.random(n=n_samples)
        l_bounds = [-1] * n_params
        u_bounds = [1] * n_params
        parameters = qmc.scale(sample, l_bounds, u_bounds)
        parameters = self.Q.germ2variable(parameters)
        return parameters

    def save_model(self, name=None, path=None):
            '''
            Saves the data-driven model to a file or returns a byte stream.
    
            Parameters:
            -----------
            name : str, optional
                Name of the file to save the model (default is None).
            path : str, optional   
                Directory path to save the file (default is None).
    
            Returns:
            -------
            None or bytes
                If name is provided, saves the model to a file. Otherwise, returns a byte stream.        
            '''
    
            if name is not None:
                name = name + ".msic"
                if path is not None:
                    name = os.path.join(path, name)
                with open(name, 'wb') as file:
                    pickle.dump(self, file)
                print(f"Model saved to {name}")
            else:
                return pickle.dumps(self)

def load_model(path: str) -> Mosaic:
    assert os.path.isfile(path), "File does not exist"
    assert path[-5:] == ".msic", "File extention is not correct. Select a '.msic' file"
    with open(path, 'rb') as handle:
        model = pickle.load(handle)
    return model
