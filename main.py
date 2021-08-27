import numpy as np
from system_param import SystemParam, create_random_system
from sensor import SensorMessage, Sensor, RandomSensor
from estimator import Estimator, Estimator2, Estimator3
from plot import plot_traj


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

        # this should be deleted after delta fix
        if k != 0:
            # delta = gamma_u[k - 1] * (1 - e[k - 1]) + (1 - gamma_u[k - 1]) * e[k - 1]
            delta = gamma_u[k - 1]
        else:
            delta = 1

        # send code
        msg = sensor.send_code()

        # update user
        if gamma_u[k] == 0:
            user.update(None)
            # user.update(None, delta)  # it should not know delta
        else:
            user.update(msg)

        # update eavesdropper
        if gamma_e[k] == 0:
            eavesdropper.update(None)  # it should not know delta
        else:
            eavesdropper.update(msg)

        # send acknowledgement
        delta = gamma_u[k] * (1 - e[k]) + (1 - gamma_u[k]) * e[k]
        sensor.update_reference_time(delta)


def test():
    dim = 1
    num_steps = 100

    params = create_random_system(dim=dim, stable=True)

    # sensor = Sensor(params)
    sensor = RandomSensor(params, probability_send_state=0.1)

    user = Estimator3(params)
    eavesdropper = Estimator3(params)

    lambda_u = 0.5
    lambda_e = 0.5
    p = 0.1

    gamma_u = np.random.binomial(1, lambda_u, num_steps)
    gamma_e = np.random.binomial(1, lambda_e, num_steps)
    e = np.random.binomial(1, p, num_steps)
    run_sim(sensor, user, eavesdropper, num_steps, gamma_u, gamma_e, e)
    # print(sensor.x_trajectory)
    # print(sensor.w_trajectory)
    # print(sensor.a_trajectory)
    plot_traj(sensor, user, eavesdropper, e)



if __name__ == '__main__':
    test()

