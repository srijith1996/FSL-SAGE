import matplotlib.pyplot as plt
import logging

def plot_final_metrics(
    tr_loss_list, tr_acc_list, tr_grad_mse_list, tr_grad_nmse_list,
    tr_loss, tr_acc, ts_loss, ts_acc,
    plot_dir, debug=False
):

    # doing this to avoid getting too many debug statements
    # from matplotlib
    old_log_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    save = (plot_dir is not None)

    # for training data
    if debug:
        fig, ax = plt.subplots(2, 2, figsize=[8, 6], dpi=120)
        legs = []
        for i in range(len(tr_loss_list)):
            ax[0, 0].semilogy(tr_grad_mse_list[i], alpha=0.8, lw=0.5)
            ax[0, 0].set_ylabel('MSE of grad (aux vs. serv)')

            ax[0, 1].semilogy(tr_grad_nmse_list[i], alpha=0.8, lw=0.5)
            ax[0, 1].set_ylabel('NMSE of grad (aux vs. serv)')
            #ax[0, 1].set_xlim([0, 200])

            ax[1, 0].plot(tr_loss_list[i], alpha=0.6)
            ax[1, 0].set_xlabel('client iter $k$')
            ax[1, 0].set_ylabel('Loss on tr. set')

            ax[1, 1].plot(tr_acc_list[i], alpha=0.6)
            ax[1, 1].set_xlabel('client iter $k$')
            ax[1, 1].set_ylabel('Accuracy on tr. set')

            legs.append(f"Client #{i}")

        ax[0, 0].legend(legs)

        for ax_ in ax.flatten():
            ax_.grid(True, which='both', axis='both')

        plt.tight_layout()
        if save: fig.savefig(f"{plot_dir}/train_per_client.png")

    # for training data
    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    ax.plot(tr_loss)
    ax.set_xlabel("Round")
    ax.set_ylabel("Training Loss")
    ax.grid(True, which='both', axis='both')
    plt.tight_layout()
    if save: fig.savefig(f"{plot_dir}/train_loss.png")

    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    ax.plot(tr_acc)
    ax.set_xlabel("Round")
    ax.set_ylabel("Training Accuracy")
    ax.grid(True, which='both', axis='both')
    plt.tight_layout()
    if save: fig.savefig(f"{plot_dir}/train_acc.png")

    # for test data
    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    ax.plot(ts_loss)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Loss")
    ax.grid(True, which='both', axis='both')
    plt.tight_layout()
    if save: fig.savefig(f"{plot_dir}/test_loss.png")

    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    ax.plot(ts_acc)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, which='both', axis='both')
    plt.tight_layout()
    if save: fig.savefig(f"{plot_dir}/test_acc.png")

    logging.getLogger().setLevel(old_log_level)
