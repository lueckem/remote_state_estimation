import numpy as np
from system_param import SystemParam
from sensor import RandomSensor, ThresholdSensor
from estimator import Estimator
from main import run_sim


def find_optimal_alpha(params, M, lambda_u, lambda_e, p, alpha_range=(0, 1), tol=1e-2, max_steps=np.infty):
    """

    Parameters
    ----------
    params : SystemParam
    M : float
    lambda_u : float
    lambda_e : float
    p : float
    alpha_range : tuple[float], optional
    tol : float, optional
        tolerance for |M - error_u|
    max_steps : float, optional
        maximum number of bisection steps

    Returns
    -------
    alpha : float
    error_u : float
    error_e : float
    """
    steps_per_run = 50000
    runs_per_iteration = 1

    up = alpha_range[1]
    low = alpha_range[0]
    alpha = 0.5 * (low + up)
    error_u = np.infty
    error_e = np.infty
    this_step = 0

    while np.abs(error_u - M) > tol:
        print("(low, alpha, up): ({},{},{})".format(np.round(low, 4), np.round(alpha, 4), np.round(up, 4)))
        this_step += 1
        if this_step > max_steps:
            print("Reached maximum number of bisection steps.")
            break
        error_u = np.infty
        error_e = np.infty
        counter = 0
        dif = np.infty
        this_tol = tol
        while dif > this_tol:
            counter += 1
            print(counter)
            this_error_u = []
            this_error_e = []
            for _ in range(runs_per_iteration):
                sensor = RandomSensor(params, probability_send_state=alpha)
                user = Estimator(params, state_update=False)
                eavesdropper = Estimator(params, state_update=False)
                gamma_u = np.random.binomial(1, lambda_u, steps_per_run)
                gamma_e = np.random.binomial(1, lambda_e, steps_per_run)
                e = np.random.binomial(1, p, steps_per_run)
                run_sim(sensor, user, eavesdropper, steps_per_run, gamma_u, gamma_e, e)
                this_error_u.append(user.mean_error)
                this_error_e.append(eavesdropper.mean_error)

            new_error_u = np.mean(this_error_u) if counter == 1 else\
                (1 - 1 / counter) * error_u + (1 / counter) * np.mean(this_error_u)
            dif = np.abs(new_error_u - error_u)
            error_u = new_error_u
            this_tol = max(10 ** np.floor(np.log10(np.abs(error_u - M))), tol / 10)
            error_e = np.mean(this_error_e) if counter == 1 else\
                (1 - 1 / counter) * error_e + (1 / counter) * np.mean(this_error_e)

        print("user error: {}".format(np.round(error_u, 4)))
        print("eaves error: {}".format(np.round(error_e, 4)))
        print("")
        if error_u < M:
            up = alpha
        else:
            low = alpha
        alpha = 0.5 * (low + up)

    return alpha, error_u, error_e


def find_optimal_threshold(params, M, lambda_u, lambda_e, p, threshold_range=(0, 1), tol=1e-2):
    """

    Parameters
    ----------
    params : SystemParam
    M : float
    lambda_u : float
    lambda_e : float
    p : float
    threshold_range : tuple[float], optional
    tol : float, optional
        tolerance for |M - error_u|

    Returns
    -------
    alpha : float
    error_u : float
    error_e : float
    """
    steps_per_run = 50000
    runs_per_iteration = 1

    up = threshold_range[1]
    low = threshold_range[0]
    threshold = 0.5 * (low + up)
    error_u = np.infty
    error_e = np.infty

    while np.abs(error_u - M) > tol:
        print("(low, threshold, up): ({},{},{})".format(np.round(low, 4), np.round(threshold, 4), np.round(up, 4)))
        error_u = np.infty
        error_e = np.infty
        counter = 0
        dif = np.infty
        this_tol = tol
        while dif > this_tol:
            counter += 1
            print(counter)
            this_error_u = []
            this_error_e = []
            for _ in range(runs_per_iteration):
                sensor = ThresholdSensor(params, threshold, lambda_u, p)
                user = Estimator(params, state_update=False)
                eavesdropper = Estimator(params, state_update=False)
                gamma_u = np.random.binomial(1, lambda_u, steps_per_run)
                gamma_e = np.random.binomial(1, lambda_e, steps_per_run)
                e = np.random.binomial(1, p, steps_per_run)
                run_sim(sensor, user, eavesdropper, steps_per_run, gamma_u, gamma_e, e)
                this_error_u.append(user.mean_error)
                this_error_e.append(eavesdropper.mean_error)
            print("max: {}".format(np.max(this_error_e)))
            print("median: {}".format(np.median(this_error_e)))

            new_error_u = np.mean(this_error_u) if counter == 1 else\
                (1 - 1 / counter) * error_u + (1 / counter) * np.mean(this_error_u)
            dif = np.abs(new_error_u - error_u)
            error_u = new_error_u
            this_tol = max(10 ** np.floor(np.log10(np.abs(error_u - M))), tol / 10)
            error_e = np.mean(this_error_e) if counter == 1 else\
                (1 - 1 / counter) * error_e + (1 / counter) * np.mean(this_error_e)

        print("user error: {}".format(np.round(error_u, 4)))
        print("eaves error: {}".format(np.round(error_e, 4)))
        print("")
        if error_u < M:
            low = threshold
        else:
            up = threshold
        threshold = 0.5 * (low + up)

    return threshold, error_u, error_e


if __name__ == '__main__':
    A = np.array([[1.2, 0.1],
                  [0, 0.5]])
    Q = np.array([[0.6, 0.2],
                  [0.2, 0.5]])
    params = SystemParam(A, Q)
    params.x0 = np.ones(2) * np.nan
    lambda_u = 0.8
    lambda_e = 0.8
    p = 0.1
    M = 2

    """ random sensor """
    # print(find_optimal_alpha(params, M, lambda_u, lambda_e, p, alpha_range=(0.3, 0.33), tol=1e-1))
    # print(0.305, 1.97, 15)

    """ threshold sensor """
    print(find_optimal_threshold(params, M, lambda_u, lambda_e, p, threshold_range=(0.17, 0.2), tol=1e-1))
    print(0.188, 1.96, 1154)
