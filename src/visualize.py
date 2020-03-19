import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class visualize:
    figsize = (15, 10)

    def __init__(self):
        pass

    def boxplot(self, df) -> None:
        """

        :param df: continuous feature data frame
        :return: all feature boxplot in one curve
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax = sns.boxplot(data=df)  # RUN PLOT
        plt.show()
        plt.clf()
        plt.close()

    def pairplots(self, df):
        """

        :param df: continuous feature data frame
        :return: pairplot to check the correlation among continuous features
        """
        return sns.pairplot(df)

    def heatmap(self, df) -> None:
        """

        :param df: continuous feature data frame
        :return: heat map to check the correlation among continuous features in qualitative way
        """
        f = plt.figure(figsize=self.figsize)
        plt.matshow(df.corr(), fignum=f.number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)

    def plot_roc(self, y_act, y_pred) -> None:
        y_pred = pd.DataFrame(y_pred)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(y_act.shape[0]):
            fpr[i], tpr[i], _ = roc_curve(y_act[i:], y_pred[i:])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_act.values.ravel(), y_pred.values.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.autoscale(enable=True, axis='both')
        plt.show()
        return "ROC curve plotted!"

    def plot_confusion_matrix(self, y_pred, y_act, label_y, label_n) -> None:
        ax = plt.subplot()
        sns.heatmap(confusion_matrix(y_act, y_pred), annot=True, ax=ax, fmt='g',
                    cmap='Greens');  # annot=True to annotate cells
        sns.set(rc={'figure.figsize': self.figsize})
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        ax.xaxis.set_ticklabels([label_n, label_y]);
        ax.yaxis.set_ticklabels([label_n, label_y]);