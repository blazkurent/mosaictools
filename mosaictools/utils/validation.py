import numpy as np
from sklearn.model_selection import KFold
from ..Mosaic import Mosaic

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


def calculate_relative_frequency_errors(real_frequencies : np.ndarray, predicted_frequencies : np.ndarray) -> np.ndarray:
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

def calculate_eigenvector_MAC_errors(real_eigenvectors : np.ndarray, predicted_eigenvectors : np.ndarray) -> np.ndarray:
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
    
    mac_errors = np.zeros((real_eigenvectors.shape[0], real_eigenvectors.shape[1]))
    for i in range(real_eigenvectors.shape[1]):
        mac_errors[:, i] = 1 - _calculate_diag_MAC_matrix(predicted_eigenvectors[:, i, :], real_eigenvectors[:, i, :])
    return mac_errors

def cross_validate(model: Mosaic, parameters: np.ndarray, frequencies: np.ndarray, eigenvectors: np.ndarray, n_folds: int = 9, shuffle: bool = False, verbose=False):

    kf = KFold(n_splits=n_folds, shuffle=shuffle)

    total_frequency_errors = np.array([])
    total_mac_errors = np.array([])

    if verbose:
        print("Cross-validation started. The process may take a couple minutes")
    
    for i, (train_index, test_index) in enumerate(kf.split(parameters)):
        train_parameters, train_frequencies, train_eigenvectors = parameters[train_index], frequencies[train_index], eigenvectors[train_index]
        test_parameters, test_frequencies, test_eigenvectors = parameters[test_index], frequencies[test_index], eigenvectors[test_index]

        model.fit(train_parameters, train_frequencies, train_eigenvectors, verbose=False)

        predicted_frequencies, predicted_eigenvectors = model.predict(test_parameters)

        frequency_errors = calculate_relative_frequency_errors(test_frequencies, predicted_frequencies)
        mac_errors = calculate_eigenvector_MAC_errors(test_eigenvectors, predicted_eigenvectors)

        if i == 0:
            total_frequency_errors = frequency_errors
            total_mac_errors = mac_errors
        else:
            total_frequency_errors = np.concatenate((total_frequency_errors, frequency_errors), axis=0)
            total_mac_errors = np.concatenate((total_mac_errors, mac_errors), axis=0)
        
        if verbose:
            if i == 0:
                print("{}/{} fold is done".format(i+1, n_folds))
            else:
                print("{}/{} folds are done".format(i+1, n_folds))
    return total_frequency_errors, total_mac_errors
