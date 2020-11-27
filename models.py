def my_scorer(estimator, X, y=None):
    from sklearn.metrics import mean_squared_error
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1*mean_squared_error(X, X_preimage)

def findbestestimator(param_grid,estimator,X,y,k=5,scoring='roc_auc'):
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(estimator = estimator, param_grid = param_grid,
                          cv = k, verbose = 2,scoring=scoring)
    clf.fit(X, y)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    return clf

def findBestNoOfComponents(X_train):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components = len(X_train[0]))
    X_train1 = pca.fit_transform(X_train)
    explained_variance = pca.explained_variance_ratio_
    #print("explained variance {}".format(explained_variance))
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print("Min no of components to retain 99.95% variance:")
    n_chosen = len(var1)+1
    for i in range(len(var1)):
        if var1[i]>=99.95:
            n_chosen=i+1
            break

    
    plt.plot(var1)
    plt.title('Varaince against no of components')
    plt.savefig("figures/pca-choosing-n")
    print(n_chosen)
    return n_chosen


def xgb(X_train,y_train,X_test):
    print("XGB")
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
    
    classifier = findbestestimator(param_grid,xgb,X_train,y_train) 
    #classifier = XGBClassifier(colsample_bytree=1,gamma=0.5, max_depth= 5, min_child_weight= 1, subsample=0.8)
    #classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def rf(X_train,y_train,X_test):
    print("rf")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=0)
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 400, 500, 800, 1000]
    }
    classifier = findbestestimator(param_grid,rf,X_train,y_train)
    #classifier = RandomForestClassifier(bootstrap=True, max_depth=80, max_features=3, min_samples_leaf=3, min_samples_split=8,n_estimators=100)
    #classifier.fit(X_train, y_train)
    #To quickly classify
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def nn(X_train,y_train,X_test):
    print("NN")
    from sklearn.neural_network import MLPClassifier
    param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['sgd', 'adam'],
            'hidden_layer_sizes': [
             (2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,)
             ]
        }
       ]
    n = MLPClassifier(random_state=0)
    classifier = findbestestimator(param_grid,n,X_train,y_train)
    #classifier = MLPClassifier(hidden_layer_sizes=(15,),activation='relu',solver='adam',random_state = 0)
    #classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return (y_pred)

def svm(X_train,y_train,X_test):
    print("SVM")
    from sklearn.svm import SVC
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],  
    #           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #           'kernel': ['rbf']}
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    s=  SVC(random_state=0)
    classifier = findbestestimator(param_grid,s,X_train,y_train)
    #classifier = SVC(C=1000,gamma=0.001,kernel='rbf',random_state=0)
    #classifier = SVC(C=10,kernel='linear')
    #classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return (y_pred)

def pca(X_train,X_test):
  from sklearn.decomposition import PCA
  n = findBestNoOfComponents(X_train)
  pca = PCA(n_components = n)
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  return (X_train,X_test)
  
def kpca(X_train,X_test):
  import numpy as np
  from sklearn.decomposition import KernelPCA
  from sklearn.model_selection import GridSearchCV
  n = findBestNoOfComponents(X_train)
  param_grid = [{
        "n_components":[n],
        "gamma": np.linspace(0.03, 0.05, 10),
        "kernel": [ "sigmoid", "poly"]
    }]
  #kpca1=KernelPCA(fit_inverse_transform=True, n_jobs=-1)
  kpca = KernelPCA(gamma=0.03,kernel='sigmoid',n_components=n)
  #kpca = GridSearchCV(kpca1, param_grid, cv=5,verbose=2, scoring=my_scorer)
  #kpca.fit(X_train)
  #print("best param",kpca.best_params_)
  X_train = kpca.fit_transform(X_train)
  X_test = kpca.transform(X_test)
  return (X_train,X_test)
  
def cal_entropy(data):
    from collections import Counter
    from math import log2
    count = Counter(data)
    items = list(count.keys())
    tot = len(list(count.elements()))
    print(tot)
    entropy=0
    for item in items:
        Class=count[item]/tot
        entropy+=-1*Class*log2(Class)
    return entropy
def discretize_feature(feature):
  import numpy as np
  mean=np.mean(feature)
  std=np.std(feature)
  discretized=np.copy(feature)
  
  discretized[np.where(feature<(mean+std/2)) ,]=2#within 1/2 std div
  discretized[np.where(feature>(mean-std/2)),]=2#within 1/2 std div
  
  discretized[np.where(feature>(mean+std/2)),]=0#greater than half
  discretized[np.where(feature<(mean-std/2)),]=1#less than half
  
  return discretized


#Function for finding the Probability distribution of single discrete variable
def pd(x):
  import numpy as np
  rX=np.unique(x)
  pX={}
  for t in rX:
    pX[t]=round(len(np.where(x==t)[0])/len(x),4)
    
  return pX
  
#Function for finding the joint probability distribution of 2 discrete variables
def pD(x,y):
  import numpy as np
  rX=np.unique(x)
  rY=np.unique(y)
  pXY={}
  pX=0
  pY_given_X=0
  
  for t1 in rX:
    i1=np.where(x==t1)[0]
    pX=len(i1)/len(x)
    
    for t2 in rY:
      pY_given_X=len(np.where(y[i1]==t2)[0])/len(y[i1])
      pXY[(t1,t2)]=round(pX*pY_given_X,4)
    
  return pXY


  
#mutual information between two discrete random variables
def mutual_info(x,y):
  import numpy as np
  x=discretize_feature(x)
  c=0
  rX=np.unique(x)
  rY=np.unique(y)
  
  pX=pd(x)
  pY=pd(y)
  pX_Y=pD(x,y)
  
  
  
  Aentropy=0.0
  for t1 in rX:
    if pX[t1]!=0:
      Aentropy-=pX[t1]* np.log2(pX[t1])
  Bentropy=0.0
  for t1 in rY:
    if pY[t1]!=0:
      Bentropy-=pY[t1]*np.log2(pY[t1])
  
  ABentropy=0.0
  for t1 in rX:
    for t2 in rY:
      if pX_Y[(t1,t2)]!=0:
        ABentropy-=pX_Y[(t1,t2)] * np.log2(pX_Y[(t1,t2)])
  #    else:
  #     print(0)
  #print(iX_Y)
  
  return (Aentropy+Bentropy-ABentropy)/(Aentropy+Bentropy) # returns normalized mutual information gain

"""
Call this function if your objective is to pass an attribute matrix or gene set and measure the Mutual Information against a 
discretized feature 'targetClass'
"""
def mutual_info_wrapper(features,targetClass):
  import numpy as np
  mi=np.array([])
  l = features.shape[1]
  for x in range(l):
    mi=np.append(mi,mutual_info(features[:][x],targetClass))
  return np.array(mi)

def info_gain(X_train,X_test,y_train):
    import pandas as pd
    ds_entropy= cal_entropy(y_train)
    print(ds_entropy)
    mutual_info_wrapper(X_train,y_train)
    X_train_df_ = pd.DataFrame(data=X_train)
    mi=mutual_info_wrapper(X_train_df_,y_train)
    quater = sum(mi)/(1.5*len(mi))
    print(quater)
    X_test_df_ = pd.DataFrame(data=X_test)
    for x in range(len(mi)):
        if mi[x]<=quater:
            X_train_df_.drop(columns=[x],axis=1,inplace=True)
            X_test_df_.drop(axis=1,columns=[x],inplace=True)

    print(X_train_df_.shape)
    return (X_train_df_,X_test_df_)


