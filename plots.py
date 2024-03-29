import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import itertools

#function that plot confusion matrix to file
def plot_confusion_matrix(cm,
                          target_names,filename,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(filename, dpi=fig.dpi)
    plt.close()

#function that plot roc curve to file
def plotRoc_curve(fpr,tpr,auc,filename,title):
    fig = plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='AUROC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig(filename, dpi=fig.dpi)
    plt.close()

#function that plot precision recall curve to file
def plotPrecisionRecall_curve(precision,recall,auc,filename,title):
    fig = plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    plt.plot(recall, precision,label='AUPRC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    fig.savefig(filename, dpi=fig.dpi)
    plt.close()
