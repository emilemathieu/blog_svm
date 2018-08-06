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
        prediction = self.intercept_ + np.sum(prod,1)
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
   
    # def _compute_kernel_matrix_diag(self, X):
    #     n_samples, n_features = X.shape
    #     diag = np.zeros(n_samples)
    #     for j,x_j in enumerate(X):
    #         diag[j] = self.kernel(x_j, x_j)
    #     return diag
        
    def _compute_intercept(self, alpha, yg):
        indices = (alpha < self.C) * (alpha > 0)
        return np.mean(yg[indices])
        
    def _compute_weights(self, X, y):
        iteration = 0
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples) #feasible solution
        g = np.ones(n_samples) #gradient initialization
        #diag = self._compute_kernel_matrix_diag(X)
        while True:

            yg = g * y
            # Working Set Selection
            indices_y_pos = (y == 1)
            indices_y_neg = (np.ones(n_samples) - indices_y_pos).astype(bool)#(y == -1)
            indices_alpha_big = (alpha >= self.C)
            indices_alpha_neg = (alpha <= 0)
            
            indices_violate_Bi_1 = indices_y_pos * indices_alpha_big
            indices_violate_Bi_2 = indices_y_neg * indices_alpha_neg
            indices_violate_Bi = indices_violate_Bi_1 + indices_violate_Bi_2
            yg_i = yg.copy()
            yg_i[indices_violate_Bi] = float('-inf') #do net select violating indices
            
            indices_violate_Ai_1 = indices_y_pos * indices_alpha_neg
            indices_violate_Ai_2 = indices_y_neg * indices_alpha_big
            indices_violate_Ai = indices_violate_Ai_1 + indices_violate_Ai_2
            yg_j = yg.copy()
            yg_j[indices_violate_Ai] = float('+inf') #do net select violating indices
            
            i = np.argmax(yg_i)
            Ki = self._compute_kernel_matrix_row(X, i)
            Kii = Ki[i]
            #indices_violate_criterion = yg_i[i] - yg <= 0
            #vec_j = (yg_i[i] - yg)**2 / (Kii - diag - 2*Ki)
            #vec_j[indices_violate_Ai_1+indices_violate_Ai_2+indices_violate_criterion] = float('-inf')
            
            j = np.argmin(yg_j)
            #j = np.argmax(vec_j)
            Kj = self._compute_kernel_matrix_row(X, j)

            # Stop criterion: stationary point or max iterations
            stop_criterion = yg_i[i] - yg_j[j] < self.tol
            if stop_criterion or (iteration >= self.max_iter and self.max_iter != -1):
                break
            
            #compute lambda
            min_1 = (y[i]==1)*self.C -y[i] * alpha[i]
            min_2 = y[j] * alpha[j] + (y[j]==-1)*self.C
            min_3 = (yg_i[i] - yg_j[j])/(Kii + Kj[j] - 2*Ki[j])
            lambda_param = np.min([min_1, min_2, min_3])
            
            #update gradient
            g = g + lambda_param * y * (Kj - Ki)
            alpha[i] = alpha[i] + y[i] * lambda_param
            alpha[j] = alpha[j] - y[j] * lambda_param
            
            iteration += 1
        # compute intercept
        intercept = self._compute_intercept(alpha, yg)
        
        print('{} iterations for gradient ascent'.format(iteration))
        return alpha, intercept