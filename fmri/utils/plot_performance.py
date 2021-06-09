import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from fmri.utils.utils import create_missing_folders

def plot_performance(loss_total,
                     kl_divs,
                     losses_recon,
                     shapes,
                     results_path,
                     filename="NoName",
                     verbose=0,
                     std_loss=None,
                     std_accuracy=None,
                     accuracies = None
                     ):
    """

    :param loss_total:
    :param loss_labelled:
    :param loss_unlabelled:
    :param accuracy:
    :param labels:
    :param results_path:
    :param filename:
    :param verbose:
    :return:
    """

    handles = []
    labels = []

    fig2, ax21 = plt.subplots(figsize=(20, 20))
    n = list(range(len(loss_total["train"])))
    try:
        ax21.plot(loss_total["Train total loss"], 'b-', label='Train total loss:' + str(len(shapes["train"])))  # plotting t, a separately
        ax21.plot(loss_total["Valid loss"], 'g-', label='Valid total loss:' + str(len(shapes["valid"])))  # plotting t, a separately
        #ax21.plot(values["valid"], 'r-', label='Test:' + str(len(labels["valid"])))  # plotting t, a separately
    except:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:')  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["train"], yerr=[np.array(std_loss["train"]), np.array(std_loss["train"])],
                      c="b", label='Train')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["valid"], yerr=[np.array(std_loss["valid"]), np.array(std_loss["valid"])],
                      c="g", label='Valid')  # plotting t, a separately

    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Loss')
    if accuracies is not None:
        ax21.plot(accuracies["train"], 'c--', label='Train accuracies')  # plotting t, a separately
        ax21.plot(accuracies["valid"], 'k--', label='Valid accuracies')  # plotting t, a separately
        if std_accuracy is not None:
            ax21.errorbar(x=n, y=accuracies["train"],
                          yerr=[np.array(std_accuracy["train"]), np.array(std_accuracy["train"])],
                          c="c", label='Train')  # plotting t, a separately
        if std_accuracy is not None:
            ax21.errorbar(x=n, y=accuracies["valid"],
                          yerr=[np.array(std_accuracy["valid"]), np.array(std_accuracy["valid"])],
                          c="k", label='Valid')  # plotting t, a separately


    if kl_divs is not None:
        ax21.plot(kl_divs["train"], 'c--', label='Train KL div')  # plotting t, a separately
        ax21.plot(kl_divs["valid"], 'k--', label='Valid KL div')  # plotting t, a separately
        if std_accuracy is not None:
            ax21.errorbar(x=n, y=kl_divs["train"], yerr=[np.array(std_accuracy["train"]), np.array(std_accuracy["train"])],
                          c="c", label='Train')  # plotting t, a separately
        if std_accuracy is not None:
            ax21.errorbar(x=n, y=kl_divs["valid"], yerr=[np.array(std_accuracy["valid"]), np.array(std_accuracy["valid"])],
                          c="k", label='Valid')  # plotting t, a separately

        ax21.plot(losses_recon["train"], 'c', label='Train Recon loss')  # plotting t, a separately
        ax21.plot(losses_recon["valid"], 'k', label='Valid Recon loss')  # plotting t, a separately
        if std_accuracy is not None:
            ax21.errorbar(x=n, y=losses_recon["train"], yerr=[np.array(std_accuracy["train"]), np.array(std_accuracy["train"])],
                          c="c", label='Train')  # plotting t, a separately
        if std_accuracy is not None:
            ax21.errorbar(x=n, y=losses_recon["valid"], yerr=[np.array(std_accuracy["valid"]), np.array(std_accuracy["valid"])],
                          c="k", label='Valid')  # plotting t, a separately

        handle, label = ax21.get_legend_handles_labels()
        handles.extend(handle)
        labels.extend(labels)

        ax21.legend(handle, label)

        fig2.tight_layout()

    create_missing_folders(results_path + "/plots/")
    pylab.savefig(results_path + "/plots/" + filename)
    plt.close()

