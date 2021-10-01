import numpy as np
from system_param import SystemParam, create_random_system
from sensor import SensorMessage, Sensor, RandomSensor, ThresholdSensor
from estimator import Estimator
from plot import plot_traj
from matplotlib import pyplot as plt


def run_sim(sensor, user, eavesdropper, num_steps, gamma_u, gamma_e, e):
    """
    Parameters
    ----------
    sensor : Sensor
    user : Estimator
    eavesdropper : Estimator
    num_steps : int
    gamma_u : np.ndarray
        shape=(num_steps,), 0=dropout
    gamma_e : np.ndarray
        shape=(num_steps,), 0=dropout
    e : np.ndarray
        shape=(num_steps,), 1=attack
    """
    for k in range(num_steps):
        # update system and sensor
        sensor.update()

        # send code
        msg = sensor.send_code()

        # update user and eavesdropper
        user.update(msg) if gamma_u[k] == 1 else user.update(None)
        eavesdropper.update(msg) if gamma_e[k] == 1 else eavesdropper.update(None)

        # send acknowledgement
        delta = gamma_u[k] * (1 - e[k]) + (1 - gamma_u[k]) * e[k]
        sensor.update_reference_time(delta)


def test():
    num_steps = 120

    # dim = 1
    # params = create_random_system(dim=dim, stable=True)

    A = np.array([[0.9]])
    Q = np.array([[1]])

    A = np.array([[1.2, 0.1],
                  [0, 0.5]])
    Q = np.array([[0.6, 0.2],
                  [0.2, 0.5]])

    params = SystemParam(A, Q)
    params = create_random_system(1, stable=False)
    lambda_u = 0.7
    lambda_e = 0.7
    alpha = 0.1
    p = 0.1

    # sensor = RandomSensor(params, probability_send_state=alpha)
    sensor = ThresholdSensor(params, 0.1, lambda_u, p)
    user = Estimator(params)
    eavesdropper = Estimator(params)

    gamma_u = np.random.binomial(1, lambda_u, num_steps)
    gamma_e = np.random.binomial(1, lambda_e, num_steps)
    e = np.random.binomial(1, p, num_steps)
    # gamma_u[23] = 0
    # e[23] = 1
    run_sim(sensor, user, eavesdropper, num_steps, gamma_u, gamma_e, e)

    # print(user.mean_error)
    # print(eavesdropper.mean_error)
    # print(sensor.x_trajectory)
    # print(sensor.w_trajectory)
    # print(sensor.a_trajectory)
    plot_traj(sensor, user, eavesdropper, e)


if __name__ == '__main__':
    test()

