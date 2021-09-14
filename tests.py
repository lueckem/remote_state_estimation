from unittest import TestCase
import numpy as np
from sensor import Sensor, RandomSensor
from estimator import Estimator
from system_param import create_random_system


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
        z, ref_time, a = self.sensor.send_code()
        self.assertEqual(z.shape, (2,))
        self.assertTrue(np.allclose(self.sensor.x[1] - self.params.L @ self.params.x0, z))
        self.assertEqual(a, 0)
        self.sensor.update_reference_time(0)

        self.sensor.update()
        z, ref_time, a = self.sensor.send_code()
        self.assertTrue(np.allclose(self.sensor.x[2] - self.params.L @ self.params.L @ self.params.x0, z))
        self.sensor.update_reference_time(1)

        self.sensor.update()
        z, ref_time, a = self.sensor.send_code()
        self.assertTrue(np.allclose(self.sensor.x[3] - self.params.L @ self.sensor.x[2], z))

        self.assertEqual(self.sensor.a_trajectory.shape, (4,))

    def test_msg_plain(self):
        self.rsensor.update()
        z, ref_time, a = self.rsensor.send_code()
        self.assertEqual(z.shape, (2,))
        self.assertTrue(np.allclose(self.rsensor.x[1], z))
        self.assertEqual(a, 1)


class TestEstimator(TestCase):
    def setUp(self):
        self.params = create_random_system(dim=2, stable=True)
        self.estimator = Estimator(self.params)
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

    def test_dropout(self):
        # test values for 2 steps
        self.rsensor.update()
        self.estimator.update(None)
        self.rsensor.update()
        self.estimator.update(None)

        x0 = self.params.x0
        A = self.params.A
        Q = self.params.Q
        x_est = [x0, A @ x0, A @ A @ x0]
        P_est = [np.zeros((self.params.dim, self.params.dim)),
                 Q,
                 A @ Q @ A.T + Q]
        self.assertTrue(np.allclose(x_est, self.estimator.x_hat))
        self.assertTrue(np.allclose(P_est, self.estimator.P))

        # test behavior in limit
        for i in range(1000):
            self.rsensor.update()
            self.estimator.update(None)
            if np.linalg.norm(self.estimator.P[-1] - self.estimator.P[-2], ord=np.inf) < 1e-8:
                break

        P_last = self.estimator.P[-1]
        x_last = self.estimator.x_hat[-1]
        self.assertTrue(np.allclose(P_last, A @ P_last @ A.T + Q))
        self.assertTrue(np.allclose(x_last, np.zeros(self.params.dim), atol=1e-3))

    def test_update_delta(self):
        # test if delta, hat_z, P_z are updated after dropouts

        # some dropouts
        for i in range(3):
            self.rsensor.update()
            self.estimator.update(None)

        # critical event
        self.rsensor.update()
        self.estimator.update(None)
        self.rsensor.update_reference_time(1)

        # some dropouts
        for i in range(5):
            self.rsensor.update()
            self.estimator.update(None)

        self.assertTrue(np.array([self.estimator.delta[k] is None for k in range(1, len(self.estimator.delta))]).all())
        self.assertTrue(np.isnan(self.estimator.z_hat_trajectory[1:, :]).all())
        self.assertTrue(np.isnan(self.estimator.Pz_trajectory[1:, :, :]).all())

        # receive code
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)

        print(self.estimator.z_hat)

        self.assertTrue(np.array([self.estimator.delta[k] is None for k in range(1, 4)]).all())
        self.assertTrue(self.estimator.delta[4] == 1)
        self.assertTrue(np.array([self.estimator.delta[k] == 0 for k in range(5, 10)]).all())
        self.assertTrue(np.isnan(self.estimator.z_hat_trajectory[1:5, :]).all())
        self.assertFalse(np.isnan(self.estimator.z_hat_trajectory[5:, :]).any())
        self.assertTrue(np.isnan(self.estimator.Pz_trajectory[1:5, :, :]).all())
        self.assertFalse(np.isnan(self.estimator.Pz_trajectory[5:, :]).any())

    def test_plain(self):
        # only send plain state -> x_hat = x
        self.rsensor.alpha = 1
        num_it = 10
        for k in range(num_it):
            self.rsensor.update()
            msg = self.rsensor.send_code()
            self.estimator.update(msg)

        self.assertTrue(np.allclose(self.rsensor.x_trajectory, self.estimator.x_hat_trajectory))
        self.assertAlmostEqual(np.max(np.abs(self.estimator.P_trajectory)), 0)

    def test_recover_from_code(self):
        # recover the correct exact from state-secrecy code in the following setting:
        # k=1: receive code (-> ref_time = 1)
        # k=2,...,n: dropouts
        # k=n+1: receive code
        self.rsensor.alpha = 0

        # k = 1
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)
        self.rsensor.update_reference_time(1)

        # k = 2,...,m
        num_it = 2
        for k in range(num_it):
            self.rsensor.update()
            self.estimator.update(None)

        # k = m + 1
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)

        self.assertTrue(np.allclose(self.rsensor.x[1], self.estimator.x_hat[1], atol=1e-3, rtol=1e-3))
        self.assertTrue(np.allclose(self.rsensor.x[-1], self.estimator.x_hat[-1], atol=1e-3, rtol=1e-3))
        self.assertTrue(np.allclose(self.estimator.P[1], np.zeros((self.params.dim, self.params.dim))))
        self.assertTrue(np.allclose(self.estimator.P[-1], np.zeros((self.params.dim, self.params.dim))))

    def test_after_critical_event(self):
        # k=1: critical event
        # k=2: receive state-secrecy code
        self.rsensor.alpha = 0

        # k=1: critical event
        self.rsensor.update()
        self.estimator.update(None)
        self.rsensor.update_reference_time(1)

        # k=2: receive state-secrecy code
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)

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
        self.assertTrue(np.allclose(self.estimator.P[-1], P_est))

    def test_after_critical_event_2(self):
        # k = 1: critical event
        # k = 2 dropout
        # k = 3: successful reception of state-secrecy code

        self.rsensor.alpha = 0

        # k=1
        self.rsensor.update()
        self.estimator.update(None)
        self.rsensor.update_reference_time(1)

        # k=2
        self.rsensor.update()
        self.estimator.update(None)

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)

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
        self.assertTrue(np.allclose(self.estimator.P[-1], P_est))

    def test_after_critical_event_3(self):
        # k = 1: dropout + critical event
        # k = 2: dropout + critical event
        # k = 3: successful reception of state-secrecy code

        self.rsensor.alpha = 0

        # k=1
        self.rsensor.update()
        self.estimator.update(None)
        self.rsensor.update_reference_time(1)

        # k=2
        self.rsensor.update()
        self.estimator.update(None)
        self.rsensor.update_reference_time(1)

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)

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
        self.assertTrue(np.allclose(self.estimator.P[-1], P_est))

    def test_after_critical_event_4(self):
        # k = 1: dropout + critical event
        # k = 2: reception of plain state, attack
        # k = 3: reception of state-secrecy code (ref_time = 1)
        self.rsensor.alpha = 0

        # k=1
        self.rsensor.update()
        self.estimator.update(None)
        self.rsensor.update_reference_time(1)

        # k=2
        self.rsensor.update()
        self.rsensor.alpha = 1
        msg = self.rsensor.send_code()
        self.rsensor.alpha = 0
        self.estimator.update(msg)

        # k=3
        self.rsensor.update()
        msg = self.rsensor.send_code()
        self.estimator.update(msg)

        A = self.params.A
        L = self.params.L
        Q = self.params.Q
        H = self.params.H
        x0 = self.params.x0
        x2 = self.rsensor.x[2]

        sigma_xx = A @ Q @ A.T + Q
        sigma_xz = A @ Q @ H.T + Q
        sigma_zz = H @ Q @ H.T + Q
        z_2 = H @ A @ x0 + sigma_xz.T @ np.linalg.inv(sigma_xx) @ (x2 - A @ A @ x0)
        Pz_2 = sigma_zz - sigma_xz.T @ np.linalg.inv(sigma_xx) @ sigma_xz
        x_est = A @ x2 + Q @ np.linalg.inv(Q + L @ Pz_2 @ L.T) @ (msg.z - H @ x2 - L @ z_2)
        P_est = Q - Q @ np.linalg.inv(Q + L @ Pz_2 @ L.T) @ Q.T

        self.assertTrue(np.allclose(self.estimator.x_hat[-1], x_est))
        self.assertTrue(np.allclose(self.estimator.P[-1], P_est))

    # def test_if_P_correct(self):
    #     num_steps = 10
    #     gamma = (np.random.random(num_steps) > 0.5).astype(int)
    #     e = (np.random.random(num_steps) > 0.9).astype(int)
    #     a = (np.random.random(num_steps) > 0.9).astype(int)
    #
    #     err = []
    #
    #     for _ in range(100):
    #         self.estimator = Estimator(self.params)
    #         self.rsensor = RandomSensor(self.params, probability_send_state=0)
    #
    #         for i in range(len(gamma)):
    #             self.rsensor.update()
    #             msg = None
    #             if gamma[i] == 1:
    #                 self.rsensor.probability_send_state = a[i]
    #                 msg = self.rsensor.send_code()
    #             self.estimator.update(msg)
    #             delta = gamma[i] * (1 - e[i]) + (1 - gamma[i]) * e[i]
    #             self.rsensor.update_reference_time(delta)
    #
    #         err.append(np.linalg.norm(self.estimator.x_hat[-1] - self.rsensor.x[-1]) ** 2)
    #
    #     print(np.mean(err), np.trace(self.estimator.P[-1]))
