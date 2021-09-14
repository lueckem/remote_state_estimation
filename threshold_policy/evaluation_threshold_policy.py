import numpy as np
from system_param import SystemParam
from sensor import ThresholdSensor
from estimator import Estimator
from matplotlib import pyplot as plt
from main import run_sim
from scipy.signal import savgol_filter


def sensor_different_threshold():
    num_steps = 200000

    A = np.array([[0.9]])
    Q = np.array([[1]])
    L = np.array([[1.0 / 0.9]])

    params = SystemParam(A, Q, L)
    lambda_u = 0.8
    lambda_e = 0.8
    p = 0.2
    thresholds = np.linspace(0, 1, 101)

    err_u = []
    err_e = []

    for threshold in thresholds:
        print(threshold)
        sensor = ThresholdSensor(params, threshold, lambda_u, p)
        user = Estimator(params)
        eavesdropper = Estimator(params)

        gamma_u = np.random.binomial(1, lambda_u, num_steps)
        gamma_e = np.random.binomial(1, lambda_e, num_steps)
        e = np.random.binomial(1, p, num_steps)
        run_sim(sensor, user, eavesdropper, num_steps, gamma_u, gamma_e, e)

        err_u.append(user.mean_error)
        err_e.append(eavesdropper.mean_error)

    np.save("thresholds4.npy", thresholds)
    np.save("erru4.npy", err_u)
    np.save("erre4.npy", err_e)


def plot_eval():
    # lambda_u, lambda_e, p
    # 1: 0.8, 0.8, 0.1
    # 2: 0.6, 0.8, 0.1
    # 3: 0.8, 0.6, 0.1
    # 4: 0.8, 0.8, 0.2

    thresholds1 = np.load("thresholds1.npy")
    err_u1 = np.load("erru1.npy")
    err_e1 = np.load("erre1.npy")

    thresholds2 = np.load("thresholds2.npy")
    err_u2 = np.load("erru2.npy")
    err_e2 = np.load("erre2.npy")

    thresholds3 = np.load("thresholds3.npy")
    err_u3 = np.load("erru3.npy")
    err_e3 = np.load("erre3.npy")

    thresholds4 = np.load("thresholds4.npy")
    err_u4 = np.load("erru4.npy")
    err_e4 = np.load("erre4.npy")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.plot(thresholds1, savgol_filter(np.array(err_u1) / np.array(err_e1), 51, 8), '-', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.1)$")
    # ax.plot(thresholds2, savgol_filter(np.array(err_u2) / np.array(err_e2), 51, 8), linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.6,0.8,0.1)$")
    # ax.plot(thresholds3, savgol_filter(np.array(err_u3) / np.array(err_e3), 51, 8), '--', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.6,0.1)$")
    # ax.plot(thresholds4, savgol_filter(np.array(err_u4) / np.array(err_e4), 51, 8), '-.', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.2)$")

    plt.plot(thresholds1, np.array(err_u1) / np.array(err_e1), '-', label="1")
    plt.plot(thresholds2, np.array(err_u2) / np.array(err_e2), '-', label="2")
    plt.plot(thresholds3, np.array(err_u3) / np.array(err_e3), '-', label="3")
    plt.plot(thresholds4, np.array(err_u4) / np.array(err_e4), '-', label="4")

    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon^u / \varepsilon^e$")
    ax.set_ylim([0, 2.5])
    plt.tight_layout()
    plt.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.13)
    plt.savefig("random.pdf")
    plt.show()


if __name__ == '__main__':
    # sensor_different_threshold()
    plot_eval()
