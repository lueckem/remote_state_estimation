import numpy as np
from system_param import SystemParam
from sensor import RandomSensor
from estimator import Estimator
from matplotlib import pyplot as plt
from main import run_sim
from scipy.signal import savgol_filter


def random_sensor_different_alpha():
    num_steps = 600000
    steps_per_run = 50

    A = np.array([[0.9]])
    Q = np.array([[1]])
    # L = np.array([[1.0 / 0.9]])

    A = np.array([[1.2, 0.1],
                  [0, 0.5]])
    Q = np.array([[0.6, 0.2],
                 [0.2, 0.5]])

    params = SystemParam(A, Q)
    params.x0 = np.ones(2) * np.nan
    lambda_u = 0.8
    lambda_e = 0.8
    p = 0.1
    alphas = np.linspace(0.01, 1, 31)[7:11]
    print(alphas)

    err_u = []
    err_e = []

    for alpha in alphas:
        print(alpha)
        num_its = 0
        this_err_u = []
        this_err_e = []
        while num_its < num_steps:
            num_its += steps_per_run
            sensor = RandomSensor(params, probability_send_state=alpha)
            user = Estimator(params, state_update=False)
            eavesdropper = Estimator(params, state_update=False)

            gamma_u = np.random.binomial(1, lambda_u, steps_per_run)
            gamma_e = np.random.binomial(1, lambda_e, steps_per_run)
            e = np.random.binomial(1, p, steps_per_run)
            run_sim(sensor, user, eavesdropper, steps_per_run, gamma_u, gamma_e, e)

            this_err_u.append(user.mean_error)
            this_err_e.append(eavesdropper.mean_error)
        err_u.append(np.mean(this_err_u))
        err_e.append(np.mean(this_err_e))

    np.save("Xalphas3.npy", alphas)
    np.save("Xerru3.npy", err_u)
    np.save("Xerre3.npy", err_e)


def plot_eval():
    # lambda_u, lambda_e, p
    # 1: 0.8, 0.8, 0.1
    # 2: 0.6, 0.8, 0.1
    # 3: 0.8, 0.6, 0.1
    # 4: 0.8, 0.8, 0.2

    alphas1 = np.load("alphas1.npy")
    err_u1 = np.load("erru1.npy")
    err_e1 = np.load("erre1.npy")

    alphas2 = np.load("alphas2.npy")
    err_u2 = np.load("erru2.npy")
    err_e2 = np.load("erre2.npy")

    alphas3 = np.load("alphas3.npy")
    err_u3 = np.load("erru3.npy")
    err_e3 = np.load("erre3.npy")

    alphas4 = np.load("alphas4.npy")
    err_u4 = np.load("erru4.npy")
    err_e4 = np.load("erre4.npy")

    # plt.plot(alphas1, np.array(err_u1) / np.array(err_e1), '-', label="1")
    # plt.plot(alphas2, np.array(err_u2) / np.array(err_e2), '-', label="2")
    # plt.plot(alphas3, np.array(err_u3) / np.array(err_e3), '-', label="3")
    # plt.plot(alphas4, np.array(err_u4) / np.array(err_e4), '-', label="4")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.plot(alphas1, savgol_filter(np.array(err_u1) / np.array(err_e1), 51, 8), '-', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.1)$")
    ax.plot(alphas2, savgol_filter(np.array(err_u2) / np.array(err_e2), 51, 8), linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.6,0.8,0.1)$")
    ax.plot(alphas3, savgol_filter(np.array(err_u3) / np.array(err_e3), 51, 8), '--', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.6,0.1)$")
    ax.plot(alphas4, savgol_filter(np.array(err_u4) / np.array(err_e4), 51, 8), '-.', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.2)$")

    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon^u / \varepsilon^e$")
    # ax.set_ylim([0, 2.5])
    plt.tight_layout()
    plt.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.13)
    plt.savefig("random.pdf")
    plt.show()


if __name__ == '__main__':
    # random_sensor_different_alpha()
    # plot_eval()
    alphas1 = np.load("Xalphas.npy")
    print(alphas1)
    err_u1 = np.load("Xerru.npy")
    print(err_u1)
    err_e1 = np.load("Xerre.npy")
    idx = np.array(err_u1) < 20
    # plt.plot(alphas1[idx], np.array(err_u1)[idx] / np.array(err_e1)[idx], '-x', label="1")
    plt.plot(alphas1[idx], np.array(err_u1)[idx], '-x', label="1")
    plt.plot(alphas1[idx], np.array(err_e1)[idx], '-x', label="2")
    plt.show()
