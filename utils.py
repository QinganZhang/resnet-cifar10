from cProfile import label
import matplotlib.pyplot as plt

def plot_metrics(series, labels, xlabel, ylabel, xticks, yticks,
                save_path="./output"):
    plt.figure(figsize=(8,4), dpi=600)
    for x, x_label in zip(series, labels):
        plt.plot(x, label=x_label)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.savefig(save_path, transperent=True, pad_inches=0.1)