import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.manifold import TSNE

@torch.no_grad()
def predict(model, dataloader):
    """
    Returns: numpy arrays of true labels and predicted probabilities.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    labels = []
    probs = []

    for batch_idx, batch in enumerate(dataloader):

        inputs, label = batch
        inputs = inputs.to(device)
        label = label.to(device)
        labels.append(label)
        outputs = model(inputs)
        probs.append(torch.sigmoid(outputs[:,1]))
    
    labels = torch.cat(labels).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    
    return labels, probs

def plot_roc_curve(train_labels, train_probs, test_labels, test_probs):
    train_fpr, train_tpr, thresholds = metrics.roc_curve(train_labels, train_probs)
    test_fpr, test_tpr, thresholds = metrics.roc_curve(test_labels, test_probs)
    fig, axes = plt.subplots()
    axes.plot(train_fpr, train_tpr, label='ROC AUC (training) = {:.2f}'.format(metrics.auc(train_fpr, train_tpr)))
    axes.plot(test_fpr, test_tpr, label='ROC AUC (test) = {:.2f}'.format(metrics.auc(test_fpr, test_tpr)))
    axes.set_xlabel('FPR')
    axes.set_ylabel('TPR')
    axes.legend()
    return fig

def plot_representations(representations, labels):
    tsne = TSNE(random_state=42)
    X = tsne.fit_transform(representations)
    fig, axes = plt.subplots(figsize=(4.8, 4.8))
    axes.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', vmax=10, alpha=0.7, linewidths=0)
    axes.set_xticks([])
    axes.set_yticks([])
    return fig
