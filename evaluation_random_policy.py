import numpy as np
from system_param import SystemParam, create_random_system
from sensor import SensorMessage, Sensor, RandomSensor
from estimator import Estimator
from plot import plot_traj
from matplotlib import pyplot as plt
from main import run_sim


def random_sensor_different_alpha():
    num_steps = 50000

    A = np.array([[0.9]])
    Q = np.array([[1]])
    L = np.array([[1.0 / 0.9]])

    params = SystemParam(A, Q, L)
    lambda_u = 0.7
    lambda_e = 0.7
    p = 0.1
    alphas = np.linspace(0, 1, 51)

    err_u = []
    err_e = []

    for alpha in alphas:
        print(alpha)
        sensor = RandomSensor(params, probability_send_state=alpha)
        user = Estimator(params)
        eavesdropper = Estimator(params)

        gamma_u = np.random.binomial(1, lambda_u, num_steps)
        gamma_e = np.random.binomial(1, lambda_e, num_steps)
        e = np.random.binomial(1, p, num_steps)
        run_sim(sensor, user, eavesdropper, num_steps, gamma_u, gamma_e, e)

        err_u.append(user.mean_error)
        err_e.append(eavesdropper.mean_error)

    np.save("erru1.npy", err_u)
    np.save("erre1.npy", err_e)

    plt.plot(alphas, err_u)
    plt.plot(alphas, err_e)
    plt.grid()
    plt.show()

    plt.plot(alphas, np.array(err_u) / np.array(err_e))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    random_sensor_different_alpha()