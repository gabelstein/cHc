"""Module for classification function."""
import numpy as np
from scipy.spatial import Delaunay


from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax


class CHC(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Classification by convex hull membership.

    Classification by membership of convex hull. For each of the given classes but one, a
    convex hull is estimated. The last class is a rest class. Then, for each new point,
    the class is determined according to the convex hull in which it lies or, if it does
    not lie in any convex hull, it is determined as the rest class.

    Parameters
    ----------
    rest_class : string | int
        The class to be used as the rest class.
    hulls : dict, (default: {})
        Optional: preprocessed hulls with class label as key and scipy.spatial.Delaunay
        hull as value.

    Attributes
    ----------
    classes_ : list
        list of classes.
    """

    def __init__(self, rest_class, hulls={}):
        """Init."""
        self.rest_class = rest_class
        self.hulls = hulls

    def fit(self, X, y):
        """Fit (estimates) the hull(s) for all classes but the rest class.

        Parameters
        ----------
        X : ndarray, shape (n_trials, data_dim)
            ndarray of data
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : CHC instance
            The CHC instance.
        """

        self.classes_ = np.unique(y)
        hull_classes = np.delete(self.classes_, np.where(self.classes_ == self.rest_class))

        if self.hulls == {}:
            self.hulls = {cl: Delaunay(X[y == cl]) for cl in hull_classes}

        return self

    def predict(self, X_test):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, data_dim)
            ndarray of test data.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to membership in hull.
        """
        # TODO: make decision what to decide when two class memberships
        pred = np.ones((X_test.shape[0]))*self.rest_class
        for cl, hull in self.hulls.items():
            pred[hull.find_simplex(X_test) >= 0] = cl

        return pred

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-self.predict(X))