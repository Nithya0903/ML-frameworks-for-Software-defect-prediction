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

def nn(X_train,y_train,X_test):
    from sklearn.neural_network import MLPClassifier
    param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,)
             ]
        }
       ]
    n = MLPClassifier(random_state=0)
    #classifier = findbestestimator(param_grid,n,X_train,y_train)
    classifier = MLPClassifier(hidden_layer_sizes=(19,),activation='relu',solver='adam',random_state = 0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def svm(X_train,y_train,X_test):
    from sklearn.svm import SVC
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
    s=  SVC(random_state=0)
    #classifier = findbestestimator(param_grid,s,X_train,y_train)
    classifier = SVC(C=1,gamma=1,kernel='rbf',random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred

def pca(X_train,y_train):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 21)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    print("explained variance {}".format(explained_variance))

    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print(var1)
    plt.plot(var1)
    plt.show()
    