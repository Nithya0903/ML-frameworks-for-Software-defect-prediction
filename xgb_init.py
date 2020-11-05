def xgb(X_train,y_train,X_test,y_test):
    # Fitting XGB-Classifier to the training set
    from xgboost import XGBClassifier
    classifier = XGBClassifier(iterations = 10000, learning_rate = 0.1,random_state=0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

    ####################################################################

    