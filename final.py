import sys
from preprocesser import preprocess
from plotter import plotter
from xgb_init import xgb
from metrics import evaluate
def main():
    X_train,X_test,y_test,y_train =preprocess(sys.argv[1])
    y_pred = xgb(X_train,y_train,X_test,y_test)
    plotter(y_test,y_pred)
    cm,fscore,a=evaluate(y_test,y_pred)

    


main()