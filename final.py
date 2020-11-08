import sys
from preprocesser import preprocess
from plotter import plotter
from models import xgb,rf,nn,svm,pca
from metrics import evaluate

def runAllModels(X_train,X_test,y_test,y_train):
    # y_pred= xgb(X_train,y_train,X_test)
    # plotter(y_test,y_pred)
    # cm,fscore,a=evaluate(y_test,y_pred)
    # y_pred = rf(X_train,y_train,X_test)
    # plotter(y_test,y_pred)
    # cm,fscore,a=evaluate(y_test,y_pred)
    # y_pred = nn(X_train,y_train,X_test)
    # plotter(y_test,y_pred)
    # cm,fscore,a=evaluate(y_test,y_pred)
    y_pred = svm(X_train,y_train,X_test)
    plotter(y_test,y_pred)
    cm,fscore,a=evaluate(y_test,y_pred)
    


def main():
    X_train,X_test,y_test,y_train =preprocess(sys.argv[1])
    runAllModels(X_train,X_test,y_test,y_train)
    X_train,X_test= pca(X_train,X_test)
    print(len(X_train[0]))
    runAllModels(X_train,X_test,y_test,y_train)

    


main()