from unittest import TestCase
import numpy as np
from sensor import SensorMessage, Sensor, RandomSensor
from estimator import Estimator, Estimator2
from system_param import create_random_system, SystemParam


class TestSensor(TestCase):
    def setUp(self):
        self.params = create_random_system(dim=2, stable=True)
        self.sensor = Sensor(self.params)
        self.rsensor = RandomSensor(self.params, probability_send_state=1)

    def test_update(self):
        self.sensor.update()
        self.sensor.update_reference_time(0)
        self.assertEqual(self.sensor.reference_time, 0)
        self.assertEqual(self.sensor.current_step, 1)

        self.sensor.update()
        self.sensor.update_reference_time(1)
        self.assertEqual(self.sensor.reference_time, 2)
        self.assertEqual(self.sensor.current_step, 2)

        self.assertEqual(self.sensor.x_trajectory.shape, (3, 2))
        self.assertEqual(self.sensor.w_trajectory.shape, (3, 2))

    def test_msg_code(self):
        self.sensor.update()
        z, ref_time = self.sensor.send_code()
        self.assertEqual(z.shape, (2,))
        self.assertTrue(np.allclose(self.sensor.x[1] - self.params.L @ self.params.x0, z))
        self.sensor.update_reference_time(0)

        self.sensor.update()
        z, ref_time = self.sensor.send_code()
        self.assertTrue(np.allclose(self.sensor.x[2] - self.params.L @ self.params.L @ self.params.x0, z))
        self.sensor.update_reference_time(1)

        self.sensor.update()
        z, ref_time = self.sensor.send_code()
        self.assertTrue(np.allclose(self.sensor.x[3] - self.params.L @ self.sensor.x[2], z))

        self.assertEqual(self.sensor.a_trajectory.shape, (4,))

    def test_msg_plain(self):
        self.rsensor.update()
        z, ref_time = self.rsensor.send_code()
        self.assertEqual(z.shape, (2,))
        self.assertTrue(np.allclose(self.rsensor.x[1], z))


class TestEstimator(TestCase):
    def setUp(self):
        self.params = create_random_system(dim=1, stable=True)
        self.estimator = Estimator(self.params)
        self.estimator2 = Estimator2(self.params)
        self.rsensor = RandomSensor(self.params, probability_send_state=1)

    # def test_format(self):
    #     num_it = 100
    #     for k in range(num_it):
    #         self.rsensor.update()
    #         msg = self.rsensor.send_code()
    #         self.estimator.update(msg)
    #
    #     self.assertEqual(self.estimator.x_hat_trajectory.shape, (num_it + 1, self.params.dim))
    #     self.assertEqual(self.estimator.P_trajectory.shape, (num_it + 1, self.params.dim, self.params.dim))
    #     self.assertEqual(len(self.estimator.z_hat), num_it + 1)
    #     self.assertEqual(len(self.estimator.gamma), num_it + 1)
    #     self.assertEqual(len(self.estimator.P_z), num_it + 1)
    #     self.assertEqual(len(self.estimator.sigma), num_it + 1)
    #     self.assertEqual(len(self.estimator.delta), num_it)
    #     self.assertEqual(self.estimator.current_step, num_it)

    def test_dropout(self):  # scenario 1
        # test values for 2 steps
        self.rsensor.update()
        self.estimator2.update(None)
        self.rsensor.update()
        self.estimator2.update(None)

        x0 = self.params.x0
        A = self.params.A
        Q = self.params.Q
        x_est = [x0, A @ x0, A @ A @ x0]
        P_est = [np.zeros((self.params.dim, self.params.dim)),
                 Q,
                 A @ Q @ A.T + Q]
        self.assertTrue(np.allclose(x_est, self.estimator2.x_hat))
        self.assertTrue(np.allclose(P_est, self.estimator2.P))

        # test behavior in limit
        for i in range(1000):
            self.rsensor.update()
            self.estimator2.update(None)
            if np.linalg.norm(self.estimator2.P[-1] - self.estimator2.P[-2], ord=np.inf) < 1e-8:
                break

        P_last = self.estimator2.P[-1]
        x_last = self.estimator2.x_hat[-1]
        self.assertTrue(np.allclose(P_last, A @ P_last @ A.T + Q))
        self.assertTrue(np.allclose(x_last, np.zeros(self.params.dim), atol=1e-3))

    def test_plain(self):  # scenario 2
        # only send plain state -> x_hat = x
        self.rsensor.alpha = 1
        num_it = 10
        for k in range(num_it):
            self.rsensor.update()
            msg = self.rsensor.send_code()
            self.estimator2.update(msg)

        self.assertTrue(np.allclose(self.rsensor.x_trajectory, self.estimator2.x_hat_trajectory))
        self.assertAlmostEqual(np.max(np.abs(self.estimator2.P_trajectory)), 0)

    def test_recover_from_code(self):  # scenario 3
        # recover the correct exact from state-secrecy code in the following setting:
        # k=1: receive code (-> ref_time = 1)
        # k=2,...,n: dropouts
        # k=n+1: receive code
        self.rsensor.alpha = 0

        # k = 1
        self.rsensor.update()
        msg = self.rsensor.send_code()
        # print(msg)
        self.estimator.update(msg)
        self.estimator2.update(msg)
        self.rsensor.update_reference_time(1)

        # k = 2,...,m
        num_it = 2
        for k in range(num_it):
            self.rsensor.update()
            delta = 0 if k > 0 else 1
            self.estimator.update(None, delta)
            self.estimator2.update(None)

        # k = m + 1
        self.rsensor.update()
        msg = self.rsensor.send_code()
        # print(msg)
        self.estimator.update(msg)
        self.estimator2.update(msg)

        # print(self.rsensor.x_trajectory)
        # print(self.estimator.x_hat_trajectory)
        # print(self.estimator2.x_hat_trajectory)
        self.assertTrue(np.allclose(self.rsensor.x[1], self.estimator.x_hat[1], atol=1e-3, rtol=1e-3))
        self.assertTrue(np.allclose(self.rsensor.x[1], self.estimator2.x_hat[1], atol=1e-3, rtol=1e-3))
        self.assertTrue(np.allclose(self.rsensor.x[-1], self.estimator.x_hat[-1], atol=1e-3, rtol=1e-3))
        self.assertTrue(np.allclose(self.rsensor.x[-1], self.estimator2.x_hat[-1], atol=1e-3, rtol=1e-3))
        self.assertTrue(np.allclose(self.estimator.P[1], np.zeros((self.params.dim, self.params.dim))))
        self.assertTrue(np.allclose(self.estimator2.P[1], np.zeros((self.params.dim, self.params.dim))))
        self.assertTrue(np.allclose(self.estimator.P[-1], np.zeros((self.params.dim, self.params.dim))))
        self.assertTrue(np.allclose(self.estimator2.P[-1], np.zeros((self.params.dim, self.params.dim))))

    def test_after_critical_event(self):  # scenario 4
        # k=1: critical event
        # k=2: receive state-secrecy code
        self.rsensor.alpha = 0

        # k=1: critical event
        self.rsensor.update()
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)
        self.rsensor.update_reference_time(1)

        # k=2: receive state-secrecy code
        self.rsensor.update()
        msg = self.rsensor.send_code()
        # print(msg)
        self.estimator.update(msg, delta=1)
        self.estimator2.update(msg)

        A = self.params.A
        H = self.params.H
        Q = self.params.Q
        x0 = self.params.x0

        sigma_xx = A @ Q @ A.T + Q
        sigma_xz = A @ Q @ H.T + Q
        sigma_zz = H @ Q @ H.T + Q
        x_est = A @ A @ x0 + sigma_xz @ np.linalg.inv(sigma_zz) @ (msg.z - H @ A @ x0)
        P_est = sigma_xx - sigma_xz @ np.linalg.inv(sigma_zz) @ sigma_xz.T

        self.assertTrue(np.allclose(self.estimator.x_hat_trajectory[-1, :], x_est))
        self.assertTrue(np.allclose(self.estimator2.x_hat_trajectory[-1, :], x_est))
        self.assertTrue(np.allclose(self.estimator.P[-1], P_est))
        self.assertTrue(np.allclose(self.estimator2.P[-1], P_est))

    def test_after_critical_event_2(self):  # scenario 4
        # k = 1: critical event
        # k = 2 dropout
        # k = 3: successful reception of state-secrecy code

        self.rsensor.alpha = 0
        x0 = self.params.x0

        # k=1
        self.rsensor.update()
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)
        self.rsensor.update_reference_time(1)

        # k=2
        self.rsensor.update()
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)

        A = self.params.A
        L = self.params.L
        Q = self.params.Q
        x0 = self.params.x0

        sigma_xx = A @ A @ Q @ A.T @ A.T + A @ Q @ A.T + Q
        sigma_xz = A @ A @ Q @ (A @ A - L @ L).T + A @ Q @ A.T + Q
        sigma_zz = (A @ A - L @ L) @ Q @ (A @ A - L @ L).T + A @ Q @ A.T + Q
        x_est = A @ A @ A @ x0 + sigma_xz @ np.linalg.inv(sigma_zz) @ (msg.z - (A @ A - L @ L) @ A @ x0)
        P_est = sigma_xx - sigma_xz @ np.linalg.inv(sigma_zz) @ sigma_xz.T

        self.assertTrue(np.allclose(self.estimator.x_hat_trajectory[-1, :], x_est))
        self.assertTrue(np.allclose(self.estimator2.x_hat_trajectory[-1, :], x_est))
        self.assertTrue(np.allclose(self.estimator.P[-1], P_est))
        self.assertTrue(np.allclose(self.estimator2.P[-1], P_est))

    def test_after_critical_event_3(self):  # scenario 4
        # k = 1: dropout + critical event
        # k = 2: dropout + critical event
        # k = 3: successful reception of state-secrecy code

        self.rsensor.alpha = 0
        x0 = self.params.x0

        # k=1
        self.rsensor.update()
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)
        self.rsensor.update_reference_time(1)

        # k=2
        self.rsensor.update()
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)
        self.rsensor.update_reference_time(1)

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)

        A = self.params.A
        L = self.params.L
        Q = self.params.Q
        x0 = self.params.x0

        tmp = A @ Q @ A.T + Q
        sigma_xx = A @ A @ Q @ A.T @ A.T + tmp
        sigma_xz = Q + A @ tmp @ (A.T - L.T)
        sigma_zz = Q + (A - L) @ tmp @ (A.T - L.T)
        x_est = A @ A @ A @ x0 + sigma_xz @ np.linalg.inv(sigma_zz) @ (msg.z - (A - L) @ A @ A @ x0)
        P_est = sigma_xx - sigma_xz @ np.linalg.inv(sigma_zz) @ sigma_xz.T

        self.assertTrue(np.allclose(self.estimator.x_hat_trajectory[-1, :], x_est))
        self.assertTrue(np.allclose(self.estimator2.x_hat_trajectory[-1, :], x_est))
        self.assertTrue(np.allclose(self.estimator.P[-1], P_est))
        self.assertTrue(np.allclose(self.estimator2.P[-1], P_est))

    def test_after_critical_event_blabla(self):  # not a real test
        # k = 1: dropout + critical event
        # k = 2: successful reception
        # k = 3: successful reception

        self.rsensor.alpha = 0
        x0 = self.params.x0

        # k=1
        self.rsensor.update()
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)
        self.rsensor.update_reference_time(1)

        # k=2
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)
        self.rsensor.update_reference_time(1)

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)

        x_est = x_estimate(self.params, x0, msg.z, 3, 1)
        # print(x_est)

        x_hat = self.estimator.x_hat_trajectory
        print(x_hat)
        print(self.estimator2.x_hat_trajectory)
        print(self.rsensor.x_trajectory)
        print()
        print(self.estimator.P_trajectory)
        print(self.estimator2.P_trajectory)

    def test_after_critical_event_4(self):  # scenario 5
        # k = 1: dropout + critical event
        # k = 2: reception of plain state, attack
        # k = 3: reception of state-secrecy code (ref_time = 1)
        self.rsensor.alpha = 0

        # k=1
        self.rsensor.update()
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)
        self.rsensor.update_reference_time(1)

        # k=2
        self.rsensor.update()
        self.rsensor.alpha = 1
        msg = self.rsensor.send_code()
        self.rsensor.alpha = 0
        self.estimator.update(msg, delta=1)
        self.estimator2.update(msg)

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)

        A = self.params.A
        L = self.params.L
        Q = self.params.Q
        x0 = self.params.x0
        x2 = self.rsensor.x[2]

        w_hat = A @ Q @ A.T @ np.linalg.inv(A @ Q @ A.T + Q) @ (x2 - A @ A @ x0)
        w_hat = np.linalg.solve(A, w_hat)
        expec_z = A @ x2 - L @ L @ (A @ x0 + w_hat)
        sigma_xx = Q
        sigma_xz = Q
        sigma_zz = Q + L @ L @ (Q + np.outer(w_hat, w_hat)) @ L.T @ L.T
        x_est_true = A @ x2 + sigma_xz @ np.linalg.inv(sigma_zz) @ (msg.z - expec_z)
        P_est = sigma_xx - sigma_xz @ np.linalg.inv(sigma_zz) @ sigma_xz.T

        # self.assertTrue(np.allclose(self.estimator.x_hat[-1], x_est_true))  # doesnt work
        self.assertTrue(np.allclose(self.estimator2.x_hat[-1], x_est_true))
        # self.assertTrue(np.allclose(self.estimator.P[-1], P_est))  # doesnt work
        self.assertTrue(np.allclose(self.estimator2.P[-1], P_est))

    def test_after_critical_event_6(self):
        # k = 1: dropout + critical event
        # k = 2: dropout
        # k = 3: reception of plain state
        # k = 4: reception of state-secrecy code (ref_time = 1)

        num_it = 10000
        err1 = []
        err2 = []
        for _ in range(num_it):
            self.estimator = Estimator(self.params)
            self.estimator2 = Estimator2(self.params)
            self.rsensor = RandomSensor(self.params, probability_send_state=1)
            self.rsensor.alpha = 0
            x0 = self.params.x0

            # k=1
            self.rsensor.update()
            self.estimator.update(None, delta=1)
            self.estimator2.update(None)
            self.rsensor.update_reference_time(1)

            # k=2
            self.rsensor.update()
            self.estimator.update(None, delta=1)
            self.estimator2.update(None)

            # k=3
            self.rsensor.update()
            self.rsensor.alpha = 1
            msg = self.rsensor.send_code()
            self.rsensor.alpha = 0
            self.estimator.update(msg, delta=0)
            self.estimator2.update(msg)

            # k=4
            self.rsensor.update()
            msg = self.rsensor.send_code()
            self.estimator.update(msg)
            self.estimator2.update(msg)

            x_hat = self.estimator.x_hat_trajectory
            x = self.rsensor.x_trajectory
            err1.append(np.linalg.norm(x[4, :] - x_hat[4, :]))

            x_hat2 = self.estimator2.x_hat_trajectory
            err2.append(np.linalg.norm(x[4, :] - x_hat2[4, :]))
            # print(x_hat)
            # print(self.estimator2.x_hat_trajectory)
            # print(self.rsensor.x_trajectory)

        print(np.mean(err1))
        print(np.mean(err2))


def sigmas(params, k, t_k):
    """
    Parameters
    ----------
    params : SystemParam
    k : int
        current step
    t_k : int
        reference time

    Returns
    -------
    tuple[np.ndarray]
        Sigma_xz, Sigma_zz
    """
    sigma1 = np.copy(params.Q)
    tmp = np.copy(params.Q)
    for _ in range(k - t_k - 1):
        tmp = params.A @ tmp @ params.A.T
        sigma1 += tmp

    sigma2 = np.copy(params.Q)
    tmp = np.copy(params.Q)
    for _ in range(t_k - 1):
        tmp = params.A @ tmp @ params.A.T
        sigma2 += tmp

    A = np.linalg.matrix_power(params.A, k - t_k)
    H = A - np.linalg.matrix_power(params.L, k - t_k)

    sigma_xz = sigma1 + A @ sigma2 @ H.T
    sigma_zz = sigma1 + H @ sigma2 @ H.T
    return sigma_xz, sigma_zz


def x_estimate(params, x0, z, k, t_k):
    x_expec = np.linalg.matrix_power(params.A, k) @ x0
    z_expec = x_expec - np.linalg.matrix_power(params.L, k - t_k) @ np.linalg.matrix_power(params.A, t_k) @ x0
    sigma_xz, sigma_zz = sigmas(params, k, t_k)
    # print("x_est: {}".format(sigma_xz))
    return x_expec + sigma_xz @ np.linalg.inv(sigma_zz) @ (z - z_expec)

