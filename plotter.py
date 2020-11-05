import numpy as np
import pandas as pd
def plotter(y_test,y_pred):
    from sklearn.preprocessing import label_binarize
    # y = label_binarize(y, classes=[0, 1])
    # n_classes = y.shape[1]
    n_classes=2
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],     remainder='passthrough')
    #y = np.array(columnTransformer.fit_transform(y),dtype=np.int)

    y_test = label_binarize(y_test, classes=[0, 1])
    y_test = np.array(columnTransformer.fit_transform(y_test),dtype=np.int)

    y_pred = label_binarize(y_pred, classes=[0, 1])
    y_pred = np.array(columnTransformer.fit_transform(y_pred),dtype=np.int)

    # Compute ROC curve and ROC area for each class
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    fig = plt.figure("fig -xgb_init")
    lw = 2
    plt.plot(fpr[0], tpr[0], color='red',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic cm1_PCA')
    plt.legend(loc="lower right")
    plt.savefig("figures/xgb_init")
    #plt.show()



