import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_learning_curve(history, filename,batch_size,learning_rate):
    plt.figure(figsize=(11, 6), dpi=100)
    plt.plot(np.array(history.history['loss'])/batch_size, 'o-', label='Training Loss')
    plt.plot(np.array(history.history['val_loss'])/batch_size, 'o:', color='r', label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Learning Curve')
    plt.suptitle(f"Batch size: {batch_size}, Learning rate: {learning_rate}", y=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss']), 10), range(1, len(history.history['loss']) + 1, 10))
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename) 
    
    
def plot_confusion_matrix(y_true, y_pred, classes, filename, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='black')
    fig.tight_layout()

    plt.savefig(filename)

