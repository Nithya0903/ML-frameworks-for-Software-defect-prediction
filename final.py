import sys
from preprocesser import preprocess
from plotter import plotter
from models import xgb,rf,nn,svm,pca,kpca,info_gain
from metrics import evaluate

def runAllModels(X_train,X_test,y_test,y_train):
    # y_pred= xgb(X_train,y_train,X_test)
    # plotter(y_test,y_pred)
    # cm,fscore,a=evaluate(y_test,y_pred)
    # y_pred = rf(X_train,y_train,X_test)
    # plotter(y_test,y_pred)
    # cm,fscore,a=evaluate(y_test,y_pred)
    y_pred = nn(X_train,y_train,X_test)
    plotter(y_test,y_pred)
    cm,fscore,a=evaluate(y_test,y_pred)
    # y_pred = svm(X_train,y_train,X_test)
    # plotter(y_test,y_pred)
    # cm,fscore,a=evaluate(y_test,y_pred)
    


def main():
    import pandas as pd
    X_train,X_test,y_test,y_train =preprocess(sys.argv[1])    
    #runAllModels(X_train,X_test,y_test,y_train)
    X_train,X_test= pca(X_train,X_test)
    #X_train,X_test= kpca(X_train,X_test)
    #X_train,X_test=info_gain(X_train,X_test,y_train)
    runAllModels(X_train,X_test,y_test,y_train)

    


main()