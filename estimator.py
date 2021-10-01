import numpy as np
from numpy.linalg import matrix_power as mpow
from system_param import SystemParam
from sensor import SensorMessage


# todo: rework update so that Sigma_xx, etc., are not calculated multiple times


class Estimator:
    def __init__(self, params, state_update=True):
        """
        Estimate the system state based on received information from the sensor.

        Parameters
        ----------
        params : SystemParam
        state_update : bool, optional
            state (x) update can be disabled to accelerate simulation of uncertainty P
        """
        self.state_update = state_update
        self.params = params

        self.x_hat = [params.x0]  # x_hat[k,:] = E[x_k | I_k]
        self.z_hat = [np.zeros(params.dim)]  # z_hat[k,:] = E[z_k | I_k]
        self.P = [np.zeros((params.dim, params.dim))]
        self.P_z = [np.zeros((params.dim, params.dim))]
        self.sigma = [np.zeros((params.dim, params.dim))]

        self.gamma = [1]  # dropouts
        self.delta = []  # acknowledgments
        self.z = [None]  # trajectory of received SensorMessages
        self.x_exact = [True]  # whether x_hat[k] = x[k]
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
            # update the past values up to k - 1
            for m in range(ref_time + 1, k):
                if not np.isnan(self.P_z[m][0, 0]):  # if != None, we dont have to update again
                    continue
                if self.z[m] is None:
                    a2, z2 = None, None
                else:
                    z2, ref2, a2 = self.z[m]
                self._update_z_Pz_sigma(m, a2, z2)

        # update x and P (always possible)
        self._update_P_and_x(k, a, z, ref_time)

        # update z, Pz, sigma to step k (not possible if dropout)
        if self.gamma[-1] == 1:
            # update step k
            self._update_z_Pz_sigma(k, a, z)

    def _update_P_and_x(self, k, a, z, ref_time):
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
        ref_time : int or None
        """
        if self.gamma[k] == 0:  # dropout
            if self.state_update:
                self.x_hat[k] = self.params.A @ self.x_hat[k - 1]
            self.P[k] = self._Sigma_xx(k)
            self.x_exact.append(False)
        elif a == 1:  # receive plain state
            if self.state_update:
                self.x_hat[k] = np.copy(z)
            self.P[k] = np.zeros((self.params.dim, self.params.dim))
            self.x_exact.append(True)
        else:  # receive state-secrecy code
            if self.x_exact[ref_time]:  # state can be calculated from code exactly
                self.P[k] = np.zeros((self.params.dim, self.params.dim))
                if self.state_update:
                    self.x_hat[k] = self.params.A @ self.x_hat[k - 1] + \
                                    self._Sigma_xz(k) @ np.linalg.inv(self._Sigma_zz(k)) @ (z - self._z_pred(k))
                self.x_exact.append(True)
            else:
                Sigma_xx = self._Sigma_xx(k)
                Sigma_xz = self._Sigma_xz(k)
                inv_Sigma_zz = np.linalg.inv(self._Sigma_zz(k))
                self.P[k] = Sigma_xx - Sigma_xz @ inv_Sigma_zz @ Sigma_xz.T
                if self.state_update:
                    self.x_hat[k] = self.params.A @ self.x_hat[k - 1] + \
                                    self._Sigma_xz(k) @ np.linalg.inv(self._Sigma_zz(k)) @ (z - self._z_pred(k))
                self.x_exact.append(False)

    def _update_z_Pz_sigma(self, k, a, z):
        """
        Update z_hat and P_z and sigma

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
            if self.state_update:
                z_hat = np.copy(z)
            Pz = np.zeros((self.params.dim, self.params.dim))
        else:
            if self.state_update:
                z_hat = self._z_pred(k)
            Pz = self._Sigma_zz(k)

            if self.gamma[k] == 1:
                sigma_zx = self._Sigma_xz(k).T
                sigma_xx = np.linalg.inv(self._Sigma_xx(k))
                if self.state_update:
                    z_hat += sigma_zx @ sigma_xx @ (z - self.params.A @ self.x_hat[k - 1])
                Pz -= sigma_zx @ sigma_xx @ sigma_zx.T

        if self.state_update:
            self.z_hat[k] = z_hat
        self.P_z[k] = Pz

        if self.gamma[k] == 0:
            sigma = self._Sigma_xz(k)
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
