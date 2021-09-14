import numpy as np
from numpy.linalg import matrix_power as mpow
from collections import namedtuple
from system_param import SystemParam


SensorMessage = namedtuple("SensorMessage", "z ref_time a")


class Sensor:
    def __init__(self, params):
        """
        Sensor that sends state-secrecy codes.

        Parameters
        ----------
        params : SystemParam
        """
        self.params = params
        self.x = [params.x0]
        self.w = [np.zeros(self.params.dim)]  # noise
        self.a = [0]  # 1 if state was sent, 0 if code was sent
        self.reference_time = 0
        self.current_step = 0  # idx of the current step

    @property
    def x_trajectory(self):
        return np.vstack(self.x)

    @property
    def w_trajectory(self):
        return np.vstack(self.w)

    @property
    def a_trajectory(self):
        return np.array(self.a)

    def _sample_w(self):
        if self.params.dim == 1:
            w = np.random.normal(0, self.params.Q[0] ** 0.5)
        else:
            w = np.random.multivariate_normal(np.zeros(self.params.dim), self.params.Q)
        return w

    def update(self):
        """
        Update the system x <- A * x + w.
        """
        self.current_step += 1
        w = self._sample_w()
        self.x.append(self.params.A @ self.x[-1] + w)
        self.w.append(w)

    def send_code(self):
        """
        Send state-secrecy code.

        Returns
        -------
        SensorMessage
        """
        k = self.current_step
        self.a.append(0)
        t_k = self.reference_time
        return SensorMessage(self.x[-1] - mpow(self.params.L, k - t_k) @ self.x[t_k], t_k, 0)

    def update_reference_time(self, ack):
        """
        Parameters
        ----------
        ack : int
        """
        if ack == 1:
            self.reference_time = self.current_step


class RandomSensor(Sensor):
    def __init__(self, params, probability_send_state):
        """
        Sensor that follows a random transmission policy.

        Parameters
        ----------
        params : SystemParam
        probability_send_state : float
            probability to send the plain system state instead of state-secrecy code
        """
        super().__init__(params)
        self.alpha = probability_send_state

    def send_code(self):
        """
        Randomly send state-secrecy code or plain system state.

        Returns
        -------
        SensorMessage
        """
        k = self.current_step
        t_k = self.reference_time
        if np.random.binomial(1, self.alpha) == 1:
            self.a.append(1)
            return SensorMessage(self.x[-1], t_k, 1)

        self.a.append(0)
        return SensorMessage(self.x[-1] - mpow(self.params.L, k - t_k) @ self.x[t_k], t_k, 0)


class ThresholdSensor(Sensor):
    def __init__(self, params, threshold, lambda_u, p):
        """
        Sensor that transmits the plain state if the belief of critical event exceeds the threshold.

        Parameters
        ----------
        params : SystemParam
        threshold : float
        lambda_u : float
            probability of successful reception
        p : float
            attack probability
        """
        super().__init__(params)
        self.threshold = threshold
        self.belief = 0  # belief of probability of critical event
        self.c = (1 - lambda_u) * p / (lambda_u * (1 - p) + (1 - lambda_u) * p)

    def send_code(self):
        """
        Randomly send state-secrecy code or plain system state.

        Returns
        -------
        SensorMessage
        """
        k = self.current_step
        t_k = self.reference_time
        if self.belief > self.threshold:
            self.a.append(1)
            return SensorMessage(self.x[-1], t_k, 1)

        self.a.append(0)
        return SensorMessage(self.x[-1] - mpow(self.params.L, k - t_k) @ self.x[t_k], t_k, 0)

    def update_reference_time(self, ack):
        """
        Update reference time and belief.

        Parameters
        ----------
        ack : int
        """
        if ack == 1:
            self.reference_time = self.current_step
            if self.a[self.current_step] == 0:
                self.belief = self.c + self.belief * (1 - self.c)
            else:
                self.belief = self.c
