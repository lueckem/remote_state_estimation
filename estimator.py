import numpy as np
from numpy.linalg import matrix_power as mpow
from system_param import SystemParam
from sensor import SensorMessage


class Estimator:
    def __init__(self, params):
        """
        Estimate the system state based on received information from the sensor.

        Parameters
        ----------
        params : SystemParam
        """
        self.params = params

        self.x_hat = [params.x0]  # x_hat[k,:] = E[x_k | I_k]
        self.z_hat = [np.zeros(params.dim)]  # z_hat[k,:] = E[z_k | I_k]
        self.P = [np.zeros((params.dim, params.dim))]
        self.P_z = [np.zeros((params.dim, params.dim))]
        self.sigma = [np.zeros((params.dim, params.dim))]

        self.gamma = [1]  # dropouts
        self.delta = []  # acknowledgments
        self.z = [None]  # trajectory of received SensorMessages
        self.current_step = 0

    @property
    def x_hat_trajectory(self):
        return np.vstack(self.x_hat)

    @property
    def z_hat_trajectory(self):
        return np.vstack(self.z_hat)

    @property
    def P_trajectory(self):
        return np.stack(self.P, axis=0)

    @property
    def Pz_trajectory(self):
        return np.stack(self.P_z, axis=0)

    @property
    def mean_error(self):
        return np.mean(np.trace(self.P_trajectory, axis1=1, axis2=2))

    def update(self, msg):
        """
        Update x_hat and P based on the received message from the sensor.

        Parameters
        ----------
        msg : SensorMessage or None
            'None' in case of dropout.
        """
        self.current_step += 1
        k = self.current_step
        self.x_hat.append(np.full(self.params.dim, np.nan))
        self.P.append(np.full((self.params.dim, self.params.dim), np.nan))
        self.z_hat.append(np.full(self.params.dim, np.nan))
        self.P_z.append(np.full((self.params.dim, self.params.dim), np.nan))
        self.sigma.append(np.full((self.params.dim, self.params.dim), np.nan))
        self.z.append(msg)

        # unpack the message
        if msg is None:  # dropout
            self.gamma.append(0)
            z, a, ref_time = None, None, None
            self.delta.append(None)
        else:
            self.gamma.append(1)
            z, ref_time, a = msg
            self.delta.append(1 if ref_time == k - 1 else 0)

        # incorporate the info about past deltas from ref_time
        if self.gamma[-1] == 1:
            self.delta[ref_time] = 1
            self.delta[ref_time + 1:k] = [0] * (k - ref_time - 1)

        # update trajectory of z_hat, Pz, sigma
        if self.gamma[-1] == 1:
            # print(k, ref_time, len(self.delta[ref_time + 1:k]))
            # print(ref_time)
            # print(k)
            # update the past values up to k - 1
            for m in range(ref_time + 1, k):
                if self.z[m] is None:  # todo: if z_hat[m] != None, we dont have to update again
                    a2, z2 = None, None
                else:
                    z2, ref2, a2 = self.z[m]
                self._update_P_and_z(m, a2, z2)
                # print(k, m)
                self._update_sigma(m)

        # update x and P (always possible)
        self._update_P_and_x(k, a, z)

        # update z, Pz, sigma to step k (not possible if dropout)
        if self.gamma[-1] == 1:
            # update step k
            self._update_P_and_z(k, a, z)
            self._update_sigma(k)

    def _update_P_and_x(self, k, a, z):
        """
        Update x_hat and P.

        Parameters
        ----------
        k : int
            step
        a : int or None
            1 = plain state, 0 = state-secrecy code
        z : np.ndarray or None
            message
        """
        if a == 1:
            x_hat = z
            P = np.zeros((self.params.dim, self.params.dim))
        else:
            x_hat = self.params.A @ self.x_hat[k - 1]
            P = self._Sigma_xx(k)
            if self.gamma[k] == 1:
                Sigma_xz = self._Sigma_xz(k)
                inv_Sigma_zz = np.linalg.inv(self._Sigma_zz(k))
                # print(z)
                x_hat += Sigma_xz @ inv_Sigma_zz @ (z - self._z_pred(k))
                P -= Sigma_xz @ inv_Sigma_zz @ Sigma_xz.T
        self.x_hat[k] = x_hat
        self.P[k] = P

    def _update_P_and_z(self, k, a, z):
        """
        Update z_hat = E[z_k | I_k] and P_{z,k}

        Parameters
        ----------
        k : int
            step
        a : int or None
            1 = plain state, 0 = state-secrecy code
        z : np.ndarray or None
            message
        """
        if a == 0:
            z_hat = z
            Pz = np.zeros((self.params.dim, self.params.dim))
        else:
            z_hat = self._z_pred(k)
            Pz = self._Sigma_zz(k)

            if self.gamma[k] == 1:
                sigma_zx = self._Sigma_xz(k).T
                sigma_xx = np.linalg.inv(self._Sigma_xx(k))
                z_hat += sigma_zx @ sigma_xx @ (z - self.params.A @ self.x_hat[k - 1])
                Pz -= sigma_zx @ sigma_xx @ sigma_zx.T

        self.z_hat[k] = z_hat
        self.P_z[k] = Pz

    def _update_sigma(self, k):
        """
        Update sigma.

        Parameters
        ----------
        k : int
            step
        """
        if self.gamma[k] == 0:
            sigma = self.params.A @ self.P[k - 1] @ self.params.H.T + self.params.Q
            if self.delta[k - 1] == 0:
                sigma += self.params.A @ self.sigma[k - 1] @ self.params.L.T
        else:
            sigma = np.zeros((self.params.dim, self.params.dim))
        self.sigma[k] = sigma

    def _Sigma_xx(self, k):
        """
        Calculate Sigma_{k,xx}.

        Parameters
        ----------
        k : int
            step

        Returns
        -------
        np.ndarray
        """
        return self.params.A @ self.P[k - 1] @ self.params.A.T + self.params.Q

    def _Sigma_xz(self, k):
        """
        Calculate Sigma_{k,xz}.

        Parameters
        ----------
        k : int
            step

        Returns
        -------
        np.ndarray
        """
        # print(k, self.sigma)
        Sigma_xz = self.params.A @ self.P[k - 1] @ self.params.H.T + self.params.Q
        if self.delta[k - 1] == 0:
            Sigma_xz += self.params.A @ self.sigma[k - 1] @ self.params.L.T
        return Sigma_xz

    def _Sigma_zz(self, k):
        """
        Calculate Sigma_{k,zz}.

        Parameters
        ----------
        k : int
            step

        Returns
        -------
        np.ndarray
        """
        Sigma_zz = self.params.H @ self.P[k - 1] @ self.params.H.T + self.params.Q
        if self.delta[k - 1] == 0:
            tmp = self.params.H @ self.sigma[k - 1] @ self.params.L.T
            Sigma_zz += tmp + tmp.T
            Sigma_zz += self.params.L @ self.P_z[k - 1] @ self.params.L.T
        return Sigma_zz

    def _z_pred(self, k):
        """
        Calculate E[z_k | I_{k-1}].

        Parameters
        ----------
        k : int
            step

        Returns
        -------
        np.ndarray
        """
        z_pred = self.params.H @ self.x_hat[k - 1]
        if self.delta[k - 1] == 0:
            z_pred += self.params.L @ self.z_hat[k - 1]
        return z_pred
