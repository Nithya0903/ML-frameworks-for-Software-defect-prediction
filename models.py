def findbestestimator(param_grid,estimator,X,y,k=5):
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(estimator = estimator, param_grid = param_grid,
                          cv = k, verbose = 2,scoring='roc_auc')
    clf.fit(X, y)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    return clf


def xgb(X_train,y_train,X_test):
    # Fitting XGB-Classifier to the training set
    from xgboost import XGBClassifier
    xgb=XGBClassifier()
    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
    
    #classifier = findbestestimator(param_grid,xgb,X_train,y_train)
    
    
    
    classifier = XGBClassifier(colsample_bytree=0.8,gamma=5, max_depth= 4, min_child_weight= 1, subsample= 0.6)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def rf(X_train,y_train,X_test):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 400, 500, 800, 1000]
    }
    '''
    classifier = findbestestimator(param_grid,rf,X_train,y_train)
    
    '''
    classifier = RandomForestClassifier(bootstrap=True, max_depth=80, max_features=2, min_samples_leaf=3, min_samples_split=8,n_estimators=200,random_state = 0)
    classifier.fit(X_train, y_train)
    #To quickly classify
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred


    