import numpy as np
from system_param import SystemParam
from sensor import SensorMessage, Sensor, RandomSensor


def run_sim(sensor, num_steps):
    for k in range(num_steps):
        # update system and sensor
        sensor.update()

        # send code
        msg = sensor.send_code()

        # update user

        # update eavesdropper

        # send acknowledgement
        ack = 1
        sensor.update_reference_time(ack)


def test():
    dim = 2
    A = np.random.random((dim, dim))
    Q = np.random.random((dim, dim))
    Q = 0.5 * (Q + Q.T)  # make symmetric
    Q = Q + dim * np.eye(dim)  # make positive definite

    print("A stable: {}".format(np.max(np.abs(np.linalg.eigvals(A))) < 1))

    params = SystemParam(A, Q)

    sensor = Sensor(params)
    sensor = RandomSensor(params, probability_send_state=0.5)

    run_sim(sensor, 10)
    print(sensor.x_trajectory)
    print(sensor.w_trajectory)
    print(sensor.a_trajectory)


if __name__ == '__main__':
    test()

