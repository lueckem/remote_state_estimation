import numpy as np
from system_param import SystemParam, create_random_system
from sensor import SensorMessage, Sensor, RandomSensor
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
    dim = 1
    num_steps = 100

    params = create_random_system(dim=dim, stable=True)

    # sensor = Sensor(params)
    sensor = RandomSensor(params, probability_send_state=0.1)

    user = Estimator(params)
    eavesdropper = Estimator(params)

    lambda_u = 0.5
    lambda_e = 0.5
    p = 0.1

    gamma_u = np.random.binomial(1, lambda_u, num_steps)
    gamma_e = np.random.binomial(1, lambda_e, num_steps)
    e = np.random.binomial(1, p, num_steps)
    run_sim(sensor, user, eavesdropper, num_steps, gamma_u, gamma_e, e)

    print(user.mean_error)
    print(eavesdropper.mean_error)
    # print(sensor.x_trajectory)
    # print(sensor.w_trajectory)
    # print(sensor.a_trajectory)
    # plot_traj(sensor, user, eavesdropper, e)


def random_sensor_different_alpha():
    num_steps = 10000

    A = np.array([[0.9]])
    Q = np.array([[1.0]])
    L = np.array([[1.0 / 0.9]])

    params = SystemParam(A, Q, L)
    lambda_u = 0.7
    lambda_e = 0.7
    p = 0.1
    alphas = np.linspace(0, 1, 21)

    err_u = []
    err_e = []

    for alpha in alphas:
        sensor = RandomSensor(params, probability_send_state=alpha)
        user = Estimator(params)
        eavesdropper = Estimator(params)

        gamma_u = np.random.binomial(1, lambda_u, num_steps)
        gamma_e = np.random.binomial(1, lambda_e, num_steps)
        e = np.random.binomial(1, p, num_steps)
        run_sim(sensor, user, eavesdropper, num_steps, gamma_u, gamma_e, e)

        err_u.append(user.mean_error)
        err_e.append(eavesdropper.mean_error)

    print(err_u)
    print(err_e)
    plt.plot(alphas, err_u)
    plt.plot(alphas, err_e)
    plt.grid()
    plt.show()

    plt.plot(alphas, np.array(err_u) / np.array(err_e))
    plt.show()


if __name__ == '__main__':
    # test()
    random_sensor_different_alpha()

