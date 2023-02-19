import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

class ResultsOOD():

    def __init__(
        self,
        onehots: np.ndarray,
        scores: np.ndarray,
        print_metrics: bool = True,
        make_plot: bool = True,
        add_to_title: str = None
    ):
        self.onehots = onehots
        self.scores = scores
        self.print_metrics = print_metrics
        self.make_plot = make_plot
        self.add_to_title = add_to_title

    def __call__(self):
        self.get_metrics()
        if self.print_metrics:
            self._print_metrics()
        if self.make_plot:
            self.plot_results()

    def get_metrics(self):
        self.auroc = roc_auc_score(self.onehots, self.scores)

    def _print_metrics(self):
        print(f"AUROC : {round(self.auroc*100, 2)} %")

    def plot_results(self, min_value: float = None, max_value: float = None):
        plt.figure(figsize = (10,4),dpi=100)

        self.out_scores, self.in_scores = self.scores[self.onehots==1], self.scores[self.onehots==0] 

        if min_value:
            self.out_scores = self.out_scores[self.out_scores >= min_value]
            self.in_scores = self.in_scores[self.in_scores >= min_value]
        if max_value:
            self.out_scores = self.out_scores[self.out_scores <= max_value]
            self.in_scores = self.in_scores[self.in_scores <= max_value]

        if self.add_to_title is not None:
            plt.title(self.add_to_title+" AUROC="+str(float(self.auroc*100))[:6]+"%",fontsize=14)
        else:
            plt.title(" AUROC="+str(float(self.auroc*100))[:6]+"%",fontsize=14)

        vals,bins = np.histogram(self.out_scores,bins = 51)
        bin_centers = (bins[1:]+bins[:-1])/2.0

        plt.plot(bin_centers,vals,linewidth=4,color="navy",marker="",label="in test")
        plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

        vals,bins = np.histogram(self.in_scores,bins = 51)
        bin_centers = (bins[1:]+bins[:-1])/2.0

        plt.plot(bin_centers,vals,linewidth=4,color="crimson",marker="",label="out test")
        plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

        plt.xlabel("Score",fontsize=14)
        plt.ylabel("Count",fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim([0,None])

        plt.legend(fontsize = 14)

        plt.tight_layout()
        plt.show()