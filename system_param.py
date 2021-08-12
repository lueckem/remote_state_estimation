import numpy as np


class SystemParam:
    def __init__(self, A, Q, L=None, x0=None):
        """
        Container for the publicly known system parameters.

        Parameters
        ----------
        A : np.ndarray
            system matrix, shape=(n,n)
        Q : np.ndarray
            covariance matrix of noise, shape=(n,n), positive definite
        L : np.ndarray, optional
            linear code, shape=(n,n)
        x0 : np.ndarray, optional
            initial state, shape=(n,)
        """
        self.A = A
        self.Q = Q
        self.dim = A.shape[0]  # dimension of state
        if x0 is None:  # if no initial state is given, a random one is chosen
            if self.dim == 1:
                self.x0 = np.random.normal(0, Q ** 0.5)[0]
            else:
                self.x0 = np.random.multivariate_normal(np.zeros(self.dim), Q)
        else:
            self.x0 = x0

        if L is None:  # choose optimal L
            if np.max(np.abs(np.linalg.eigvals(A))) > 1:  # if A unstable
                self.L = self.A
            else:  # A stable
                L_old = Q
                L = A @ L_old @ A.T + Q
                counter = 0
                while (np.abs(L - L_old) > 1e-8).any():
                    L_old = L
                    L = A @ L_old @ A.T + Q
                    counter += 1
                    if counter > 1e5:
                        raise RuntimeError("Cannot find optimal L.")
                self.L = L @ np.linalg.inv(L @ A.T)
        else:
            self.L = L
        self.H = self.A - self.L
