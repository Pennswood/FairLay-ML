import itertools
import time
import xml_parser

import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC

def SVM(inp, X_train, X_test, y_train, y_test, sensitive_param = None):
    print(inp)
    arr, features = xml_parser.xml_parser('SVM_Params.xml',inp)
    print(arr)
    try:
        # domain-specific constraints
        if(arr[8] == 'none'):
            arr[8] = None
        # The combination of penalty='l1' and loss='hinge' is not supported, Parameters: penalty='l1', loss='hinge', dual=False
        if(arr[0] == 'l1' and arr[1] == 'hinge'):
            arr[2] = True
            arr[0] = 'l2'
        if(arr[0] == 'l2' and arr[1] == 'hinge'):
            arr[2] = True
        if(arr[0] == 'l1' and arr[1] == 'squared_hinge'):
            arr[2] = False
        arr[5] = 'ovr'
        # clf = SVC(C = arr[0], kernel= arr[1], degree= arr[2], gamma = arr[3], coef0 = arr[4],
        #         shrinking = arr[5], probability = arr[6], tol = arr[7], cache_size = arr[8],
        #         class_weight = arr[9], verbose=False, max_iter = max_itr, decision_function_shape=arr[10],
        #         break_ties = arr[11], random_state=2019)
        clf = LinearSVC(penalty = arr[0], loss = arr[1], dual = arr[2], tol = arr[3], C = arr[4],
                        multi_class = arr[5], fit_intercept = arr[6], intercept_scaling = arr[7],
                        class_weight = arr[8], verbose=0, random_state=2019, max_iter=1000)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        preds = clf.predict(X_test)
    except ValueError as ve:
        print("here you go------------------------------------")
        print(arr)
        print(ve)
        return False, None, arr, None, None, features
    print(arr)
    return True, clf, arr, score, preds, features