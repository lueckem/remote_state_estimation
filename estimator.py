import numpy as np
from system_param import SystemParam
from sensor import SensorMessage
# todo: Problem: z_hat and P_z need delta[k-1] in the case gamma[k]=0!


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
        self.gamma = [1]  # dropouts
        self.delta = []  # acknowledgments
        self.P = [np.zeros((params.dim, params.dim))]
        self.P_z = [np.zeros((params.dim, params.dim))]
        self.sigma = [np.zeros((params.dim, params.dim))]

        self.current_step = 0

    @property
    def x_hat_trajectory(self):
        return np.vstack(self.x_hat)

    @property
    def P_trajectory(self):
        return np.stack(self.P, axis=0)

    def update(self, msg, delta=None):  # delta should be removed
        """
        Update x_hat and P based on the received message from the sensor.

        Parameters
        ----------
        msg : SensorMessage or None
            'None' in case of dropout.
        """
        self.current_step += 1

        # unpack the message
        if msg is None:  # dropout
            self.gamma.append(0)
            z = None
            a = None
            self.delta.append(delta)
            # p = 0.1
            # self.delta.append((1 - p) * delta + p * (1 - delta))
        else:
            self.gamma.append(1)
            z, ref_time = msg
            a = 1 if ref_time == -1 else 0  # = gamma_s
            self.delta.append(1 if ref_time == self.current_step - 1 else 0)

        self._update_P_and_x(a, z)
        self._update_z_hat(a, z)
        self._update_P_z(a)
        self._update_sigma()

    def _update_P_and_x(self, a, z):
        k = self.current_step
        # print(k)
        # print(self.gamma)
        # print(self.delta)
        if a == 1:
            self.x_hat.append(z)
            self.P.append(np.zeros((self.params.dim, self.params.dim)))
        else:
            x_hat = self.params.A @ self.x_hat[-1]
            P = self._Sigma_xx()
            if self.gamma[k] == 1:
                Sigma_xz = self._Sigma_xz()
                inv_Sigma_zz = np.linalg.inv(self._Sigma_zz())
                # print("z_pred: {}".format(self.z_pred))
                # print("z: {}".format(self.z_hat))
                # print("x_hat: {}".format(self.x_hat))
                x_hat += Sigma_xz @ inv_Sigma_zz @ (z - self._z_pred())
                P -= Sigma_xz @ inv_Sigma_zz @ Sigma_xz.T
            self.x_hat.append(x_hat)
            self.P.append(P)

    def _Sigma_xx(self):
        k = self.current_step
        return self.params.A @ self.P[k - 1] @ self.params.A.T + self.params.Q

    def _Sigma_xz(self):
        k = self.current_step
        Sigma_xz = self.params.A @ self.P[k - 1] @ self.params.H.T + self.params.Q
        if self.delta[k - 1] == 0:
            Sigma_xz += self.params.A @ self.sigma[k - 1] @ self.params.L.T
        return Sigma_xz

    def _Sigma_zz(self):
        k = self.current_step
        Sigma_zz = self.params.H @ self.P[k - 1] @ self.params.H.T + self.params.Q
        if self.delta[k - 1] == 0:
            tmp = self.params.H @ self.sigma[k - 1] @ self.params.L.T
            Sigma_zz += tmp + tmp.T
            Sigma_zz += self.params.L @ self.P_z[k - 1] @ self.params.L.T
        return Sigma_zz

    def _z_pred(self):
        k = self.current_step
        z_pred = self.params.H @ self.x_hat[k - 1]
        if self.delta[k - 1] == 0:
            z_pred += self.params.L @ self.z_hat[k - 1]
        return z_pred

    def _update_z_hat(self, a, z):
        k = self.current_step
        if self.gamma[k] == 0:
            z_hat = self.params.H @ self.x_hat[k - 1]
            if self.delta[k - 1] == 0:
                z_hat += self.params.L @ self.z_hat[k - 1]
        elif a == 0:
            z_hat = z
        else:
            z_hat = z - self.params.L @ self.x_hat[k - 1]
            if self.delta[k - 1] == 0:
                z_hat += self.params.L @ self.z_hat[k - 1]
        self.z_hat.append(z_hat)

    def _update_P_z(self, a):
        k = self.current_step
        if self.gamma[k] == 0:
            P_z = self._Sigma_zz()
        elif a == 0:
            P_z = np.zeros((self.params.dim, self.params.dim))
        else:
            P_z = self.params.L @ self.P[k - 1] @ self.params.L.T
            if self.delta[k - 1] == 0:
                P_z += self.params.L @ (self.P_z[k - 1] + self.sigma[k - 1] + self.sigma[k - 1].T)\
                       @ self.params.L.T
        self.P_z.append(P_z)

    def _update_sigma(self):
        k = self.current_step
        if self.gamma[k] == 0:
            sigma = self.params.A @ self.P[k - 1] @ self.params.H.T + self.params.Q
            if self.delta[k - 1] == 0:
                sigma += self.params.A @ self.sigma[k - 1] @ self.params.L.T
        else:
            sigma = np.zeros((self.params.dim, self.params.dim))
        self.sigma.append(sigma)