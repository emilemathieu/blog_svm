from .binary_classes import binary_classification
import numpy as np
import itertools
from scipy.stats import mode
from sklearn.svm import SVC

def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking ovr decision function.
    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.
    Parameters
    ----------
    predictions : array-like, shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.
    confidences : array-like, shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.
    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes))
    sum_of_confidences = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1

    max_confidences = sum_of_confidences.max()
    min_confidences = sum_of_confidences.min()

    if max_confidences == min_confidences:
        return votes

    # Scale the sum_of_confidences to (-0.5, 0.5) and add it with votes.
    # The motivation is to use confidence levels as a way to break ties in
    # the votes without switching any decision made based on a difference
    # of 1 vote.
    eps = np.finfo(sum_of_confidences.dtype).eps
    max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
    scale = (0.5 - eps) / max_abs_confidence
    return votes + sum_of_confidences * scale

class Base_multiclass(object):
    def __init__(self, kernel, C):
        self.kernel = kernel
        self.C = C
        self.estimators_ = []
    
    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        """
        raise NotImplementedError()
        
    def predict(self, X):
        """Predict data's labels according to previously fitted data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Returns
        -------
        labels : array-like, shape = [n_samples]
            Predicted labels of X.
        """
        raise NotImplementedError()
        
    def score(self, X, y):
        """Compute the score of the model on X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        score : float
            Score of the model.
        """
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

class multiclass_classification(Base_multiclass):
    """ One vs One scheme
    """
    def __init__(self, kernel, C=1.0, max_iter=1000, cache_size = 200, tol=1.0):
        super().__init__(kernel, C=C)
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.tol = tol

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._pairs = list(itertools.combinations(self.classes_, 2))
        estimators = np.empty(len(self._pairs), dtype=object)
        pairs_indices = []
        for i,pair_of_labels in enumerate(self._pairs):
            print(pair_of_labels)
            SVM_binary = binary_classification(
                    kernel=self.kernel, C=self.C, max_iter=self.max_iter,
                    cache_size=self.cache_size, tol=self.tol)


            X_filtered, y_filtered, classes_indices = self._filter_dataset_by_labels(X, y, pair_of_labels)
            pairs_indices.append(classes_indices)
            SVM_binary.fit(X_filtered, y_filtered)
            estimators[i] = SVM_binary
        self.estimators_ = estimators
        self.pairs_indices = pairs_indices
        self._save_support_vectors(X)

    def _get_class_indices(self, y, label):
        return np.where(y == label)[0]
    
    def _get_classes_indices(self, y, classes):
        res = np.zeros(len(classes), len(y))
        for i, label in enumerate(classes):
            res[i,:] = self._get_class_indices(y, label)
        return res
    
    def _get_two_classes_indices(self, y, label_1, label_2):
        indices_1 = self._get_class_indices(y, label_1)
        indices_2 = self._get_class_indices(y, label_2)
        return np.concatenate((indices_1,indices_2))
    
    def _filter_dataset_by_labels(self, X, y, pair_of_labels):
        label_1, label_2 = pair_of_labels
        classes_indices = self._get_two_classes_indices(y, label_1, label_2)
        X_filtered = X[classes_indices, :]
        y_filtered = y[classes_indices]
        y_filtered[y_filtered == label_1] = -1
        y_filtered[y_filtered == label_2] = 1
        return X_filtered, y_filtered, classes_indices
    
    def _save_support_vectors(self, X):
        indices_to_compute = np.zeros(X.shape[0], dtype=int) 
        for j, estimator in enumerate(self.estimators_):
            indices_to_compute[self.pairs_indices[j][estimator.support_]] = 1
        self.indices_to_compute = indices_to_compute.astype(bool)
        self.all_support_vectors = X.compress(self.indices_to_compute,axis=0)

    def _pre_compute_kernel_vectors(self, X):
        print('Precompute kernel terms between test dataset and support vectors')
        kernel_matrix = np.zeros((X.shape[0], len(self.indices_to_compute)))
        for i,x_i in enumerate(X):
            if i % 500 == 0:
                print('{} / {}'.format(i,X.shape[0]))
            k = 0
            for j, need_compute in enumerate(self.indices_to_compute):
                if need_compute:
                    kernel_matrix[i, j] = self.kernel(x_i, self.all_support_vectors[k,:])
                    k += 1

        kernel_matrices = []
        for j, estimator in enumerate(self.estimators_):
            kernel_matrices.append(kernel_matrix[:,self.pairs_indices[j][estimator.support_]])
        return kernel_matrices
    
    def predict(self, X):
        kernel_matrices = self._pre_compute_kernel_vectors(X)
        confidences = np.empty((X.shape[0], len(self._pairs)))
        for j, estimator in enumerate(self.estimators_):
            confidences[:,j] = estimator._predict_proba(X,kernel_matrices[j])
        predictions = ((np.sign(confidences) + 1 ) / 2).astype(int)
        Y = _ovr_decision_function(predictions, confidences, len(self.classes_))    
        return self.classes_[Y.argmax(axis=1)]

    def predict_by_voting(self, X):
        """A voting scheme is applied: all K (K − 1) / 2 classifiers are applied to an unseen
        sample and the class that got the highest number of "+1" predictions gets predicted by
        the combined classifier.
        """
        n_samples = X.shape[0]
        predicted_labels = np.zeros((n_samples, len(self._pairs)))
        for j, pair_of_labels in enumerate(self._pairs):
            print(pair_of_labels)
            binary_prediction = self.estimators_[j].predict(X)
            binary_prediction[binary_prediction == -1] = pair_of_labels[0]
            binary_prediction[binary_prediction == 1] = pair_of_labels[1]
            predicted_labels[:,j] = binary_prediction
        prediction = np.ravel(mode(predicted_labels, axis=1).mode)
        return prediction