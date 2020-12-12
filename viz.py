import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ex: pmilnce-seed_64-candidate_1-train_metrics
candidate_lst = [1,4,7]
seed_lst = [42, 64]
loss_type_lst = ["milnce", "pmilnce"]
val_metric_lst = ["R1", "R5", "R10", "MR"]

def load_pickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def create_path(loss_type, seed, candidate, exp="train"):
    return "data/log/" + loss_type + "-seed_" + str(seed) + "-candidate_"+ str(candidate) + "-" + exp + "_metrics.pickle"

def plot_metric(metric_type, seed, candidate):
    milnce_metric = load_pickle(create_path("milnce", seed, candidate))
    pmilnce_metric = load_pickle(create_path("pmilnce", seed, candidate))

    if metric_type == "loss":
        milnce_x = [epoch for epoch, loss, lr in milnce_metric]
        milnce_y = [loss for epoch, loss, lr in milnce_metric]
        pmilnce_x = [epoch for epoch, loss, lr in pmilnce_metric]
        pmilnce_y = [loss for epoch, loss, lr in pmilnce_metric]

    if metric_type == "learning rate":
        milnce_x = [epoch for epoch, loss, lr in milnce_metric]
        milnce_y = [lr for epoch, loss, lr in milnce_metric]
        pmilnce_x = [epoch for epoch, loss, lr in pmilnce_metric]
        pmilnce_y = [lr for epoch, loss, lr in pmilnce_metric]


    plt.plot(milnce_x, milnce_y, label="milnce" + str(candidate))
    plt.plot(pmilnce_x, pmilnce_y, label="pmilnce" + str(candidate))

    plt.xlabel("epochs")
    plt.ylabel(metric_type)
    # plt.ion()
    plt.legend()
    plt.title(metric_type + " for candidate=" + str(candidate) +" and seed=" + str(seed))
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join("data/fig/train", metric_type + "_candidate=" + str(candidate)+ "_seed="+ str(seed) +".png"))
    plt.clf()

def plot_metric_individual(loss_type, metric_type, seed, candidate):
    metric = load_pickle(create_path(loss_type, seed, candidate))

    if metric_type == "loss":
        x = [epoch for epoch, loss, lr in metric]
        y = [loss for epoch, loss, lr in metric]

    if metric_type == "learning rate":
        x = [epoch for epoch, loss, lr in metric]
        y = [lr for epoch, loss, lr in metric]

    plt.plot(x, y, label=loss_type + " " + metric_type)

    plt.xlabel("epochs")
    plt.ylabel(metric_type)
    plt.ion()
    plt.legend()
    plt.title(loss_type + " " + metric_type + " for candidate=" + str(candidate) +" and seed=" + str(seed))
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join("data/fig/train", loss_type + "_" + metric_type + "_candidate=" + str(candidate)+ "_seed="+ str(seed) +".png"))
    plt.clf()

def get_val_scores(metric):
    x = [epoch for epoch, mydict in metric]
    r1 = [mydict["R1"] for epoch, mydict in metric]
    r5 = [mydict["R5"] for epoch, mydict in metric]
    r10 = [mydict["R10"] for epoch, mydict in metric]
    mr = [mydict["MR"] for epoch, mydict in metric]
    return x, [r1, r5, r10, mr]

def plot_val_progress(seed, candidate):
    # for each metric do a diff plot
    milnce_metric = load_pickle(create_path("milnce", seed, candidate, exp="val"))
    pmilnce_metric = load_pickle(create_path("pmilnce", seed, candidate, exp="val"))

    milnce_x, milnce_scores = get_val_scores(milnce_metric)
    pmilnce_x, pmilnce_scores = get_val_scores(pmilnce_metric)

    for i in range(4):
        metric_type = val_metric_lst[i]
        milnce_y = milnce_scores[i]
        pmilnce_y = pmilnce_scores[i]
        plt.plot(milnce_x, milnce_y, label="milnce" + " " + metric_type)
        plt.plot(pmilnce_x, pmilnce_y, label="pmilnce" + " " + metric_type)

        plt.xlabel("epochs")
        plt.ylabel(metric_type)
        plt.ion()
        plt.legend()
        plt.title("Validation " + metric_type + " for candidate=" + str(candidate) +" and seed=" + str(seed))
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join("data/fig/val", "valmetric=" + metric_type + "_candidate=" + str(candidate)+ "_seed="+ str(seed) +".png"))
        plt.clf()

def plot_val_bars(candidate, bar_metric_lst):
    # epoch 90 average over seeds
    # each two bars belong to one metric
    milnce_42 = load_pickle(create_path("milnce", 42, candidate, exp="val"))[-1][1]
    pmilnce_42 = load_pickle(create_path("pmilnce", 42, candidate, exp="val"))[-1][1]

    milnce_64 = load_pickle(create_path("milnce", 64, candidate, exp="val"))[-1][1]
    pmilnce_64 = load_pickle(create_path("pmilnce", 64, candidate, exp="val"))[-1][1]

    milnce_averaged = {}
    pmilnce_averaged = {}

    for metric in bar_metric_lst:
        milnce_averaged[metric] = (milnce_42[metric] + milnce_64[metric]) / 2.0
        pmilnce_averaged[metric] = (pmilnce_42[metric] + pmilnce_64[metric]) / 2.0


    labels = bar_metric_lst
    milnce_averaged_lst = list(milnce_averaged.values())
    pmilnce_averaged_lst = list(pmilnce_averaged.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, milnce_averaged_lst, width, label='milnce')
    rects2 = ax.bar(x + width/2, pmilnce_averaged_lst, width, label='pmilnce')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Averaged (over seeds) Final Epoch Validation Scores for Candidate=' + str(candidate))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join("data/fig/val", "averaged_valscores-bar=" + bar_metric_lst[0] + "-candidate=" + str(candidate) +".png"))
    plt.clf()

def main():
    for seed in seed_lst:
        for candidate in candidate_lst:
            plot_metric("loss", seed, candidate)
            plot_metric("learning rate", seed, candidate)

    for loss_type in loss_type_lst:
        for seed in seed_lst:
            for candidate in candidate_lst:
                plot_metric_individual(loss_type, "loss", seed, candidate)
                plot_metric_individual(loss_type, "learning rate", seed, candidate)

    for seed in seed_lst:
        for candidate in candidate_lst:
            plot_val_progress(seed, candidate)

    for candidate in candidate_lst:
        plot_val_bars(candidate, ["R1", "R5", "R10"])
        plot_val_bars(candidate, ["MR"])

if __name__ == '__main__':
    main()
