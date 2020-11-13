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
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.
    """

    def __init__(self, rest_class, hulls={}):
        """Init."""
        self.rest_class = rest_class
        self.hulls_ = hulls

    def fit(self, X, y):
        """Fit (estimates) the hull(s).

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
        self.hulls_ = {cl: Delaunay(X[y == cl]) for cl in hull_classes}

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
            the prediction for each trials according to the closest centroid.
        """
        # todo make decision what to decide when two class memberships
        pred = np.ones((X_test.shape[0]))*self.rest_class
        for cl, hull in self.hulls_.items():
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