def homework(): 
    # for exercise 1 in Colab7_Model_Selection, ported to the following function 
    import pandas as pd
    import numpy as np  
    from sklearn import datasets
    from sklearn.model_selection import GridSearchCV

    X,y = datasets.load_breast_cancer(return_X_y=True)
    X.shape, y.shape
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    mTR = DecisionTreeClassifier()
    mNN = MLPClassifier()
    mKNN = KNeighborsClassifier()
    mSVM = SVC()

    parTR = {'criterion':['gini', 'entropy'], 'max_depth':[2,3,4,5,6,7,8,9,10]}
    parSVM = {'kernel':['linear', 'rbf'], 'C':[1,2,3,4]}
    parNN = {'hidden_layer_sizes':[(10,),(20,),(30,),(40,)], 'max_iter': [1000]}
    parKNN = {'n_neighbors':[2,3,4,5]}

    model = [mTR,mNN,mKNN,mSVM]
    para = [parTR,parNN,parKNN,parSVM]
    modelName = ['Decision Tree', 'Neural Network', 'KNN', 'SVM']

    m = 4
    gSA = [0]*m
    for i in range(0, m):
        gSA[i] = GridSearchCV(model[i], para[i], cv = 5)
        gSA[i].fit(X,y);

    for i in range(0, m):
        print(gSA[i].best_score_)
    return gSA



if __name__ == '__main__':
    print(homework())
