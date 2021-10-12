import numpy as np
from system_param import SystemParam
from sensor import ThresholdSensor
from estimator import Estimator
from matplotlib import pyplot as plt
from main import run_sim
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def plot_eval():
    # lambda_u, lambda_e, p
    # 1: 0.8, 0.8, 0.1
    # 2: 0.6, 0.8, 0.1
    # 3: 0.8, 0.6, 0.1
    # 4: 0.8, 0.8, 0.2

    """ load data """
    directory = "../random_policy/"
    ran_err_u1 = np.load(directory + "erru1.npy")
    ran_err_e1 = np.load(directory + "erre1.npy")

    ran_err_u2 = np.load(directory + "erru2.npy")
    ran_err_e2 = np.load(directory + "erre2.npy")

    ran_err_u3 = np.load(directory + "erru3.npy")
    ran_err_e3 = np.load(directory + "erre3.npy")

    ran_err_u4 = np.load(directory + "erru4.npy")
    ran_err_e4 = np.load(directory + "erre4.npy")

    ran_err_u2D = np.load(directory + "Xerru.npy")
    ran_err_e2D = np.load(directory + "Xerre.npy")

    directory = "../threshold_policy/"
    err_u1 = np.load(directory + "erru1.npy")
    err_e1 = np.load(directory + "erre1.npy")
    err_u1_wrongp = np.load(directory + "erru1_wrongp.npy")
    err_e1_wrongp = np.load(directory + "erre1_wrongp.npy")

    err_u2 = np.load(directory + "erru2.npy")
    err_e2 = np.load(directory + "erre2.npy")

    err_u3 = np.load(directory + "erru3.npy")
    err_e3 = np.load(directory + "erre3.npy")

    err_u4 = np.load(directory + "erru4.npy")
    err_e4 = np.load(directory + "erre4.npy")

    err_u2D = np.load(directory + "Xerru.npy")
    err_e2D = np.load(directory + "Xerre.npy")

    err_u = [err_u1, err_u2, err_u3, err_u4, err_u1_wrongp, err_u2D]
    err_e = [err_e1, err_e2, err_e3, err_e4, err_e1_wrongp, err_e2D]
    ran_err_u = [ran_err_u1, ran_err_u2, ran_err_u3, ran_err_u4, ran_err_u1, ran_err_u2D]
    ran_err_e = [ran_err_e1, ran_err_e2, ran_err_e3, ran_err_e4, ran_err_e1, ran_err_e2D]

    """ sort and interpolate """
    x_list = []
    y_list = []

    for i in range(len(err_e)):
        idx = np.argsort(err_u[i])
        err_u[i] = err_u[i][idx]
        err_e[i] = err_e[i][idx]
        err_e[i] = err_e[i][err_u[i] < 10]
        err_u[i] = err_u[i][err_u[i] < 10]
        # print(err_u[i][:10])
        # print(err_e[i][:10])

        idx = np.argsort(ran_err_u[i])
        ran_err_u[i] = ran_err_u[i][idx]
        ran_err_e[i] = ran_err_e[i][idx]

        x_min = max(np.min(ran_err_u[i]), np.min(err_u[i]))
        x_max = min(np.max(ran_err_u[i]), np.max(err_u[i]))
        # print(x_min, x_max)
        x = np.linspace(x_min, x_max, 101)
        pol = interp1d(err_u[i], err_e[i])
        plt.plot(x, pol(x))
        ran_pol = interp1d(ran_err_u[i], ran_err_e[i])
        plt.plot(x, ran_pol(x))
        plt.show()
        y_list.append(pol(x) / ran_pol(x))
        x_list.append(x)

    """ plotting """
    label_list = [r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.1)$",
                  r"$(\lambda_u,\lambda_e,p)=(0.6,0.8,0.1)$",
                  r"$(\lambda_u,\lambda_e,p)=(0.8,0.6,0.1)$",
                  r"$(\lambda_u,\lambda_e,p)=(0.8,0.8,0.2)$", "wrongp", "2D"]
    linestyle_list = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', '-', '-']
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for i in range(len(x_list) - 1):
        ax.plot(x_list[i], savgol_filter(y_list[i], 21, 6), linewidth=2.5, linestyle=linestyle_list[i], label=label_list[i])
    # ax.plot(x, savgol_filter(y1 / y2, 51, 8), '-', label="1.5")
    # plt.plot(err_u1_wrongp, err_e1_wrongp, 'x-', label="threshold wrongp")
    # plt.plot(ran_err_u1, ran_err_e1, 'x-', label="random")
    # plt.plot(x, y)
    # plt.plot(thresholds2, np.array(err_u2) / np.array(err_e2), '-', label="2")
    # plt.plot(thresholds3, np.array(err_u3) / np.array(err_e3), '-', label="3")
    # plt.plot(thresholds4, np.array(err_u4) / np.array(err_e4), '-', label="4")
    ax.plot(x_list[-1], y_list[-1], linewidth=2.5, linestyle=linestyle_list[-1], marker='x', label=label_list[-1])

    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$\varepsilon^e_{thresh} / \varepsilon^e_{rand}$")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([0, 10])
    plt.tight_layout()
    plt.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.13)
    plt.savefig("random.pdf")
    plt.show()


if __name__ == '__main__':
    plot_eval()
