import numpy as np
from sklearn.svm import SVC
import copy


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
            y_pred = np.zeros((len(X), ))
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
