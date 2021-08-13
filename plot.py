from matplotlib import pyplot as plt
import numpy as np
from sensor import Sensor
from estimator import Estimator


def plot_traj(sensor, user, eavesdropper, e):
    """
    Parameters
    ----------
    sensor : Sensor
    user : Estimator
    eavesdropper : Estimator
    e : np.ndarray
        shape=(num_steps,), 1=attack
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [3, 3, 1]}, figsize=(5, 5))
    # fig.suptitle(
    #     r'$\lambda_u={}, \lambda_e={}, p={}, \alpha={}$'.format(lambda_u, lambda_e, lambda_a, lambda_s))
    # first subplot
    sensor_plot, = ax1.plot(sensor.x_trajectory[:, 0], '-s', label=r'$x$', color='tab:blue')
    user_plot, = ax1.plot(user.x_hat_trajectory[:, 0], '-o', label=r'$\hat{x}^u$', color='tab:green')
    eaves_plot, = ax1.plot(eavesdropper.x_hat_trajectory[:, 0], '-x', label=r'$\hat{x}^e$', color='tab:orange')
    num_steps = len(sensor.x_trajectory[:, 0])
    ax1.set_xticks(np.arange(0, num_steps, 5))
    ax1.set_xticklabels(np.arange(0, num_steps + 1, 5))
    ax1.set_xticks(np.arange(0.5, num_steps + 1, 1), minor=True)
    ax1.grid(which='minor', axis='x')
    ax1.grid(which='major', axis='y')
    ax1.legend(handles=[sensor_plot, user_plot, eaves_plot])
    # second subplot
    P_u = [np.trace(P) for P in user.P_trajectory]
    P_e = [np.trace(P) for P in eavesdropper.P_trajectory]
    user_plot, = ax2.plot(P_u, '-o', label='$P^u$', color='tab:green')
    eaves_plot, = ax2.plot(P_e, '-x', label='$P^e$', color='tab:orange')
    ax2.set_xticks(np.arange(0, num_steps, 5))
    ax2.set_xticklabels(np.arange(0, num_steps + 1, 5))
    ax2.set_xticks(np.arange(0.5, num_steps + 1, 1), minor=True)
    ax2.grid(which='minor', axis='x')
    ax2.grid(which='major', axis='y')
    ax2.legend(handles=[user_plot, eaves_plot])
    # third
    binary_array = np.zeros((4, num_steps + 1))
    binary_array[0, 1:] = user.gamma  # gamma_u
    binary_array[1, 1:] = eavesdropper.gamma  # gamma_e
    # binary_array[2, 2:] = e
    # binary_array[3, 1:] = sensor.a_trajectory
    ax3.imshow(binary_array, aspect='auto', cmap=plt.cm.get_cmap('binary'), interpolation=None)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels([r'$\gamma^u$', r'$\gamma^e$', r'$e$', r'$a$'])
    ax3.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5], minor=True)
    ax3.grid(which='minor')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig('traj.png', dpi=300)
    plt.show()
