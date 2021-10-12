import numpy as np
from system_param import SystemParam
from sensor import ThresholdSensor
from estimator import Estimator
from matplotlib import pyplot as plt
from main import run_sim
from scipy.signal import savgol_filter


def sensor_different_threshold():
    num_steps = 600000
    steps_per_run = 50


    A = np.array([[0.9]])
    Q = np.array([[1]])
    L = np.array([[1.0 / 0.9]])

    A = np.array([[1.2, 0.1],
                  [0, 0.5]])
    Q = np.array([[0.6, 0.2],
                  [0.2, 0.5]])

    params = SystemParam(A, Q)
    params.x0 = np.ones(2) * np.nan
    lambda_u = 0.8
    lambda_e = 0.8
    p = 0.1
    thresholds = np.linspace(0, 0.3, 31)[13:]
    # thresholds = [0.0855, 0.095, 0.1045, 0.114, 0.1235, 0.133, 0.1425, 0.152, 0.1615, 0.171, 0.1805, 0.19, 0.1995, 0.209]
    # thresholds = [0.209]

    err_u = []
    err_e = []

    for threshold in thresholds:
        print(threshold)
        num_its = 0
        this_err_u = []
        this_err_e = []
        while num_its < num_steps:
            num_its += steps_per_run
            sensor = ThresholdSensor(params, threshold, lambda_u, p=p)
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

    np.save("Xthresholds2.npy", thresholds)
    np.save("Xerru2.npy", err_u)
    np.save("Xerre2.npy", err_e)


def plot_eval():
    # lambda_u, lambda_e, p
    # 1: 0.8, 0.8, 0.1
    # 2: 0.6, 0.8, 0.1
    # 3: 0.8, 0.6, 0.1
    # 4: 0.8, 0.8, 0.2

    thresholds1 = np.load("thresholds1.npy")
    err_u1 = np.load("erru1.npy")
    err_e1 = np.load("erre1.npy")
    err_u1_wrongp = np.load("erru1_wrongp.npy")
    err_e1_wrongp = np.load("erre1_wrongp.npy")

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

    # ax.plot(thresholds1, savgol_filter(np.array(err_u1) / np.array(err_e1), 51, 8), '-', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.1)$")
    # ax.plot(thresholds2, savgol_filter(np.array(err_u2) / np.array(err_e2), 51, 8), linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.6,0.8,0.1)$")
    # ax.plot(thresholds3, savgol_filter(np.array(err_u3) / np.array(err_e3), 51, 8), '--', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.6,0.1)$")
    # ax.plot(thresholds4, savgol_filter(np.array(err_u4) / np.array(err_e4), 51, 8), '-.', linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.2)$")

    idx = np.concatenate((np.arange(40), np.arange(40, 91, 2), np.arange(90, 101)))

    ax.plot(thresholds1[idx], (np.array(err_u1) / np.array(err_e1))[idx], '-', linewidth=2.5,
            label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.1)$")
    ax.plot(thresholds2[idx], (np.array(err_u2) / np.array(err_e2))[idx], linestyle=(0, (3, 1, 1, 1, 1, 1)),
            linewidth=2.5, label=r"$(\lambda_u,\lambda_e,p)=(0.6,0.8,0.1)$")
    ax.plot(thresholds3[idx], (np.array(err_u3) / np.array(err_e3))[idx], '--', linewidth=2.5,
            label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.6,0.1)$")
    ax.plot(thresholds4[idx], (np.array(err_u4) / np.array(err_e4))[idx], '-.', linewidth=2.5,
            label=r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.2)$")
    ax.plot(thresholds1[idx], (np.array(err_u1_wrongp) / np.array(err_e1_wrongp))[idx], '-', linewidth=2.5,
            label=r"mismatched p")

    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\varepsilon^u / \varepsilon^e$")
    ax.set_ylim([0, 2.5])
    plt.tight_layout()
    plt.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.13)
    plt.savefig("random.pdf")
    plt.show()


if __name__ == '__main__':
    # sensor_different_threshold()
    # plot_eval()

    alphas1 = np.load("Xthresholds.npy")
    print(alphas1)
    err_u1 = np.load("Xerru.npy")
    err_e1 = np.load("Xerre.npy")
    err_u2 = np.load("Xerru2.npy")
    err_e2 = np.load("Xerre2.npy")
    idx = np.array(err_u1) < 20
    print(np.array(err_u1)[idx])
    print(np.array(err_e1)[idx])
    plt.plot(alphas1[idx], np.array(err_u1)[idx], '-x', label="1")
    plt.plot(alphas1[idx], np.array(err_e1)[idx], '-x', label="1")
    # plt.semilogy(alphas1[idx], np.array(err_u1)[idx], '-x', label="1")

    err_u1[13:] = (err_u1[13:] + err_u2) / 2
    err_e1[13:] = (err_e1[13:] + err_e2) / 2
    plt.plot(alphas1[idx], np.array(err_u1)[idx], '-o', label="1")
    plt.plot(alphas1[idx], np.array(err_e1)[idx], '-o', label="1")
    plt.plot()
    plt.show()
