import numpy as np
from numpy.linalg import matrix_power as mpow
from system_param import SystemParam
from sensor import SensorMessage


# todo: Problem: z_hat and P_z need delta[k-1] in the case gamma[k]=0!


class Estimator2:
    def __init__(self, params):
        """
        Estimate the system state based on received information from the sensor.
        Only works if params.L is chosen optimally!

        Parameters
        ----------
        params : SystemParam
        """
        self.params = params

        self.x_hat = [params.x0]  # x_hat[k,:] = E[x_k | I_k]
        self.P = [np.zeros((params.dim, params.dim))]
        self.exact = [True]  # whether we know the exact x
        self.gamma = [1]
        self.current_step = 0

    @property
    def x_hat_trajectory(self):
        return np.vstack(self.x_hat)

    @property
    def P_trajectory(self):
        return np.stack(self.P, axis=0)

    def update(self, msg):
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
            gamma_k = 0
            z, ref_time = None, None
            a = None
        else:
            gamma_k = 1
            z, ref_time = msg
            a = 1 if ref_time == -1 else 0
        self.gamma.append(gamma_k)

        # update estimates
        k = self.current_step

        if gamma_k == 0:  # dropout -- scenario 1
            self.x_hat.append(self.params.A @ self.x_hat[k - 1])
            self.P.append(self.params.A @ self.P[k - 1] @ self.params.A.T + self.params.Q)
            self.exact.append(False)
        elif a == 1:  # received plain state -- scenario 2
            self.x_hat.append(z)
            self.P.append(np.zeros((self.params.dim, self.params.dim)))
            self.exact.append(True)
        else:  # received state-secrecy code
            if self.exact[ref_time]:  # we can recover x exactly from the code -- scenario 3
                self.x_hat.append(z + mpow(self.params.L, k - ref_time) @ self.x_hat[ref_time])
                self.P.append(np.zeros((self.params.dim, self.params.dim)))
                self.exact.append(True)
            else:
                exact = np.array(self.exact)
                rev = exact[:ref_time][::-1]
                step = len(rev) - np.argmax(rev) - 1  # idx of latest exact x before ref_time
                if not (exact[ref_time:k]).any():  # scenario 4
                    # print("scenario 4")
                    x_k, P_k = self._calc_estimate_scenario4(self.x_hat[step], z, k - step, ref_time - step)
                else:  # scenario 5
                    # print("scenario 5")
                    # find the nearest exact x after ref_time
                    m = np.argmax(exact[ref_time:k]) + ref_time
                    x0 = self.x_hat[step]
                    xm = self.x_hat[m]
                    x_k, P_k = self._calc_estimate_scenario5(x0, xm, z, k - step, ref_time - step, m - step)
                self.x_hat.append(x_k)
                self.P.append(P_k)
                self.exact.append(False)

    def _calc_estimate_scenario4(self, x0, z, k, t_k):
        """
        Given the last exact state estimate is x_0, calculate the estimate for x_k and P_k.

        Parameters
        ----------
        x0 : np.ndarray
        z : np.ndarray
        k : int
        t_k : int

        Returns
        -------
        tuple[np.ndarray]
            x_k, P_k
        """
        x_expec = mpow(self.params.A, k) @ x0
        z_expec = x_expec - mpow(self.params.L, k - t_k) @ mpow(self.params.A, t_k) @ x0
        sigma_xx, sigma_xz, sigma_zz = self._sigmas_scenario4(k, t_k)
        sigma_zz = np.linalg.inv(sigma_zz)
        x_k = x_expec + sigma_xz @ sigma_zz @ (z - z_expec)
        P_k = sigma_xx - sigma_xz @ sigma_zz @ sigma_xz.T
        return x_k, P_k

    def _sigmas_scenario4(self, k, t_k):
        """
        Given the last exact state estimate is x_0, calculate Sigma_{k, xx}, Sigma_{k,xz} and Sigma_{k, zz}.

        Parameters
        ----------
        k : int
            current step
        t_k : int
            reference time, 0 < t_k < k

        Returns
        -------
        tuple[np.ndarray]
            Sigma_xx, Sigma_xz, Sigma_zz
        """
        # note: this is ineffient, same powers of A are calculated multiple times
        sigma_xx = self._sum_AQA(k)
        sigma1 = self._sum_AQA(k - t_k)
        sigma2 = self._sum_AQA(t_k)

        A = mpow(self.params.A, k - t_k)
        H = A - mpow(self.params.L, k - t_k)

        sigma_xz = sigma1 + A @ sigma2 @ H.T
        sigma_zz = sigma1 + H @ sigma2 @ H.T
        return sigma_xx, sigma_xz, sigma_zz

    def _calc_estimate_scenario5(self, x0, xm, z, k, t_k, m):
        """

        Parameters
        ----------
        x0 : np.ndarray
        xm : np.ndarray
        z : np.ndarray
        k : int
        t_k : int
        m : int

        Returns
        -------
        tuple[np.ndarray]
            x_k, P_k
        """
        # note: all these calculations are very inefficient!!!

        w_hat = [self._w_hat(x0, xm, i, m) for i in range(t_k)]

        x_expec = mpow(self.params.A, k - m) @ xm
        z_expec = np.sum([mpow(self.params.A, j - 1) @ w_hat[t_k - j] for j in range(1, t_k + 1)], axis=0)
        z_expec = x_expec - mpow(self.params.L, k - t_k) @ (mpow(self.params.A, t_k) @ x0 + z_expec)

        sigma_xx = self._sum_AQA(k - m)
        sigma_xz = sigma_xx

        sigma_zz = np.sum([mpow(self.params.A, j - 1) @ (self.params.Q + np.outer(w_hat[t_k - j], w_hat[t_k - j])) @
                           mpow(self.params.A, j - 1).T for j in range(1, t_k + 1)], axis=0)
        sigma_zz = sigma_xx + mpow(self.params.L, k - t_k) @ sigma_zz @ mpow(self.params.L, k - t_k).T

        sigma_zz = np.linalg.inv(sigma_zz)
        x_k = x_expec + sigma_xz @ sigma_zz @ (z - z_expec)
        P_k = sigma_xx - sigma_xz @ sigma_zz @ sigma_xz.T
        return x_k, P_k

    def _w_hat(self, x0, xm, k, m):
        """
        Calculate E[w_k | x0, xm] where 0 <= k < m-1

        Parameters
        ----------
        x0 : np.ndarray
        xm : np.ndarray
        k : int
        m : int

        Returns
        -------
        np.ndarray
        """
        A = mpow(self.params.A, m - k - 1)
        rhs = A @ self.params.Q @ A.T @ np.linalg.inv(self._sum_AQA(m)) @ (xm - mpow(self.params.A, m) @ x0)
        # print(xm)
        # print(mpow(self.params.A, m) @ x0)
        return np.linalg.solve(A, rhs)

    def _sum_AQA(self, k, start=1):
        """
        Calculate SUM_{j=k1}^k2 A^{j-1} Q (A^{j-1})^T

        Parameters
        ----------
        k : int
        start : int, optional

        Returns
        -------
        np.ndarray
        """
        s = mpow(self.params.A, start - 1)
        s = s @ self.params.Q @ s.T
        tmp = np.copy(s)
        for _ in range(k - start):
            tmp = self.params.A @ tmp @ self.params.A.T
            s += tmp
        return s


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
            if ref_time == -1:
                self.delta.append(delta)
            else:
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
                P_z += self.params.L @ (self.P_z[k - 1] + self.sigma[k - 1] + self.sigma[k - 1].T) \
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
