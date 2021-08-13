from unittest import TestCase
import numpy as np
from sensor import SensorMessage, Sensor, RandomSensor
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
        num_it = 2
        for k in range(num_it):
            self.rsensor.update()
            self.estimator.update(None)

        self.rsensor.update()
        msg = self.rsensor.send_code()
        print(msg)
        self.estimator.update(msg)

        print(self.rsensor.x_trajectory)
        print(self.estimator.x_hat_trajectory)
        print(self.estimator.P_trajectory)
        print(msg.z + np.linalg.matrix_power(self.params.L, 3) @ self.params.x0)
        self.assertTrue(np.allclose(self.rsensor.x[-1], self.estimator.x_hat[-1]))

