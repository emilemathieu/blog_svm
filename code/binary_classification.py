import numpy as np

class SVM(object):
    def __init__(self, kernel, C=1.0, max_iter=1000, tol=0.001):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.support_vector_tol = 0.01

    def fit(self, X, y):
        lagrange_multipliers, intercept = self._compute_weights(X, y)
        self.intercept_ = intercept
        support_vector_indices = lagrange_multipliers > self.support_vector_tol
        self.dual_coef_ = lagrange_multipliers[support_vector_indices] * y[support_vector_indices]
        self.support_vectors_ = X[support_vector_indices]

    def _compute_kernel_support_vectors(self, X):
        res = np.zeros((X.shape[0], self.support_vectors_.shape[0]))
        for i,x_i in enumerate(X):
            for j,x_j in enumerate(self.support_vectors_):
                res[i, j] = self.kernel(x_i, x_j)
        return res

    def predict(self, X):
        kernel_support_vectors = self._compute_kernel_support_vectors(X)
        prod = np.multiply(kernel_support_vectors, self.dual_coef_)
        prediction = self.intercept_ + np.sum(prod, 1)
        return np.sign(prediction)

    def score(self, X, y):
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

    def _compute_kernel_matrix_row(self, X, index):
        row = np.zeros(X.shape[0])
        x_i = X[index, :]
        for j,x_j in enumerate(X):
            row[j] = self.kernel(x_i, x_j)
        return row
   
    def _compute_intercept(self, alpha, yg):
        indices = (alpha < self.C) * (alpha > 0)
        return np.mean(yg[indices])
        
    def _compute_weights(self, X, y):
        iteration = 0
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples) # Initialise coefficients to 0  w
        g = np.ones(n_samples) # Initialise gradients to 1

        while True:
            yg = g * y

            # Working Set Selection via maximum violating constraints
            indices_y_positive = (y == 1)
            indices_y_negative = (np.ones(n_samples) - indices_y_positive).astype(bool)#(y == -1)
            indices_alpha_upper = (alpha >= self.C)
            indices_alpha_lower = (alpha <= 0)
            
            indices_violate_Bi = (indices_y_positive * indices_alpha_upper) + (indices_y_negative * indices_alpha_lower)
            yg_i = yg.copy()
            yg_i[indices_violate_Bi] = float('-inf') #cannot select violating indices
            indices_violate_Ai = (indices_y_positive * indices_alpha_lower) + (indices_y_negative * indices_alpha_upper)
            yg_j = yg.copy()
            yg_j[indices_violate_Ai] = float('+inf') #cannot select violating indices
            
            i = np.argmax(yg_i)
            j = np.argmin(yg_j)

            # Stopping criterion: stationary point or maximum iterations
            stop_criterion = yg_i[i] - yg_j[j] < self.tol
            if stop_criterion or (iteration >= self.max_iter and self.max_iter != -1):
                break
            
            # Compute lambda via Newton Method and constraints projection
            lambda_max_1 = (y[i] == 1) * self.C - y[i] * alpha[i]
            lambda_max_2 = y[j] * alpha[j] + (y[j] == -1) * self.C
            lambda_max = np.min([lambda_max_1, lambda_max_2])

            Ki = self._compute_kernel_matrix_row(X, i)
            Kj = self._compute_kernel_matrix_row(X, j)
            lambda_plus = (yg_i[i] - yg_j[j]) / (Ki[i] + Kj[j] - 2 * Ki[j])
            lambda_param = np.max([0, np.min([lambda_max, lambda_plus])])
            
            # Update gradient
            g = g + lambda_param * y * (Kj - Ki)

            # Direction search update
            alpha[i] = alpha[i] + y[i] * lambda_param
            alpha[j] = alpha[j] - y[j] * lambda_param
            
            iteration += 1

        # Compute intercept
        intercept = self._compute_intercept(alpha, yg)
        
        print('{} iterations for gradient ascent'.format(iteration))
        return alpha, intercept