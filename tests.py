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
        self.params = create_random_system(dim=2, stable=True)
        self.estimator = Estimator(self.params)
        self.estimator2 = Estimator2(self.params)
        self.rsensor = RandomSensor(self.params, probability_send_state=1)

    def test_format(self):
        num_it = 100
        for k in range(num_it):
            self.rsensor.update()
            msg = self.rsensor.send_code()
            self.estimator.update(msg)

        self.assertEqual(self.estimator.x_hat_trajectory.shape, (num_it + 1, self.params.dim))
        self.assertEqual(self.estimator.P_trajectory.shape, (num_it + 1, self.params.dim, self.params.dim))
        self.assertEqual(len(self.estimator.z_hat), num_it + 1)
        self.assertEqual(len(self.estimator.gamma), num_it + 1)
        self.assertEqual(len(self.estimator.P_z), num_it + 1)
        self.assertEqual(len(self.estimator.sigma), num_it + 1)
        self.assertEqual(len(self.estimator.delta), num_it)
        self.assertEqual(self.estimator.current_step, num_it)

    def test_plain(self):
        # only send plain state -> x_hat = x
        num_it = 100
        for k in range(num_it):
            self.rsensor.update()
            msg = self.rsensor.send_code()
            self.estimator.update(msg)

        self.assertTrue(np.allclose(self.rsensor.x_trajectory, self.estimator.x_hat_trajectory))
        self.assertAlmostEqual(np.max(np.abs(self.estimator.P_trajectory)), 0)

    def test_recover_from_code(self):
        self.rsensor.alpha = 0
        num_it = 5
        for k in range(num_it):
            self.rsensor.update()
            delta = 0 if k > 0 else 1
            self.estimator.update(None, delta)
            self.estimator2.update(None)

        self.rsensor.update()
        msg = self.rsensor.send_code()
        # print(msg)
        self.estimator.update(msg)
        self.estimator2.update(msg)

        # print(self.rsensor.x_trajectory)
        # print(self.estimator.x_hat_trajectory)
        # print(self.estimator2.x_hat_trajectory)
        # print(msg.z + np.linalg.matrix_power(self.params.L, 3) @ self.params.x0)
        self.assertTrue(np.allclose(self.rsensor.x[-1], self.estimator.x_hat[-1]))
        self.assertTrue(np.allclose(self.rsensor.x[-1], self.estimator2.x_hat[-1]))

    def test_after_critical_event(self):
        num_it = 2
        error_simple = []
        error_compl = []

        for _ in range(num_it):
            self.setUp()

            self.rsensor.alpha = 0

            # critical event at k=1
            self.rsensor.update()
            self.estimator.update(None, delta=1)
            self.rsensor.update_reference_time(1)

            self.rsensor.update()
            msg = self.rsensor.send_code()
            # print(msg)
            self.estimator.update(msg)

            x = self.rsensor.x_trajectory
            x_hat = self.estimator.x_hat_trajectory

            # print(x)
            # print(x_hat)

            # print(msg.z + self.params.L @ x[1, :])
            # print(msg.z + self.params.L @ x_hat[1, :])
            error_simple.append(np.linalg.norm(msg.z + self.params.L @ x_hat[1, :] - x[-1, :]))

            A = self.params.A
            L = self.params.L
            H = self.params.H
            Q = self.params.Q
            x0 = self.params.x0

            print(A @ x_hat[1, :] + (A @ Q @ H.T + Q) @ np.linalg.inv(H @ Q @ H.T + Q) @ (msg.z - H @ x_hat[1, :]))
            error_compl.append(np.linalg.norm(A @ x_hat[1, :] + (A @ Q @ H.T + Q) @ np.linalg.inv(H @ Q @ H.T + Q) @
                                              (msg.z - H @ x_hat[1, :]) - x[-1, :]))

        print(np.mean(error_simple))
        print(np.mean(error_compl))

    def test_after_critical_event_2(self):
        # k = 1: dropout + critical event
        # k = 2,3: dropout
        # k = 4: successful reception of state-secrecy code

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
        self.estimator.update(None, delta=0)
        self.estimator2.update(None)

        # k=4
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)

        x_est = x_estimate(self.params, x0, msg.z, 4, 1)
        print(x_est)

        x_hat = self.estimator.x_hat_trajectory
        print(x_hat)

        print(self.estimator2.x_hat_trajectory)

        # assertTrue: x_hat[4, :] == x_est

    def test_after_critical_event_3(self):
        # k = 1: dropout + critical event
        # k = 2: dropout + critical event
        # k = 3: dropout
        # k = 4: successful reception of state-secrecy code

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
        self.estimator.update(None, delta=1)
        self.estimator2.update(None)

        # k=4
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)

        x_est = x_estimate(self.params, x0, msg.z, 4, 2)
        print(x_est)

        x_hat = self.estimator.x_hat_trajectory
        print(x_hat)

        print(self.estimator2.x_hat_trajectory)
        # assertTrue: x_hat[4, :] == x_est

    def test_after_critical_event_4(self):
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

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.estimator2.update(msg)

        x_est = x_estimate(self.params, x0, msg.z, 3, 1)
        print(x_est)

        x_hat = self.estimator.x_hat_trajectory
        print(x_hat)
        print(self.estimator2.x_hat_trajectory)
        print(self.rsensor.x_trajectory)

    def test_after_critical_event_5(self):
        # k = 1: dropout + critical event
        # k = 2: reception of plain state, attack
        # k = 3: reception of state-secrecy code (ref_time = 1)

        num_it = 10000
        err1 = []
        err2 = []
        err3 = []
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

            x_hat = self.estimator.x_hat_trajectory
            x = self.rsensor.x_trajectory
            err1.append(np.linalg.norm(x[3, :] - x_hat[3, :]))

            x_hat2 = self.estimator2.x_hat_trajectory
            err2.append(np.linalg.norm(x[3, :] - x_hat2[3, :]))
            # print(x_hat)
            # print(self.estimator2.x_hat_trajectory)
            # print(self.rsensor.x_trajectory)

            A = self.params.A
            L = self.params.L
            Q = self.params.Q

            w_hat = A @ Q @ A.T @ np.linalg.inv(A @ Q @ A.T + Q) @ (x[2, :] - A @ A @ x[0, :])
            w_hat = np.linalg.solve(A, w_hat)
            expec_z = A @ x[2, :] - L @ L @ (A @ x[0, :] + w_hat)
            w_hat = w_hat.reshape((-1, 1))
            x_est_true = A @ x[2, :] + Q @ np.linalg.inv(Q + L @ L @ (Q + w_hat @ w_hat.T) @ L.T @ L.T) @ (msg.z - expec_z)
            # print(x_est_true)
            err3.append(np.linalg.norm(x[3, :] - x_est_true))

        print(np.mean(err1))
        print(np.mean(err2))
        print(np.mean(err3))


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

