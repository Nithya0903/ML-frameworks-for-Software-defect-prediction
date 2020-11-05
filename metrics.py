def evaluate(y_test,y_pred):
# Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    #f1 score
    from sklearn.metrics import f1_score
    fscore = f1_score(y_test,y_pred)
    print("F1-Score : ",fscore)

    # Finding Area Under ROC curve
    from sklearn.metrics import roc_auc_score as roc
    a = roc(y_test, y_pred, average='micro')
    print("ROC-AUC Score : ",a)
    return (cm,fscore,a)