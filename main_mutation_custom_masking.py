# Black-box fuzzer
import sys
sys.path.append("./subjects/")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
if sys.version_info.major==2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
import os
import time
import copy
from scipy.stats import randint
import csv
import argparse

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, true_positive_rate
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference

from adf_utils.config import census, credit, bank, compas
from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_data.compas import compas_data
import xml_parser
import xml_parser_domains
from Timeout import timeout

import pandas as pd
import ast
def isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help='The name of dataset: census, credit, bank ')
parser.add_argument("--algorithm", help='The name of algorithm: logistic regression, SVM, Random Forest')
parser.add_argument("--sensitive_index", help='The index for sensitive feature')
parser.add_argument("--output", help='The name of output file', required=False)
parser.add_argument("--time_out", help='Max. running time', default = 14400, required=False)
parser.add_argument("--max_iter", help='The maximum number of iterations', default = 100000, required=False)
parser.add_argument("--save_model", help='Enable save models)', default = False, required=False)
parser.add_argument("--standard_scale", help='Preprocess data with standard scaling on features before using model', default = False, required=False)
args = parser.parse_args()

def check_for_fairness(X, y_pred, y_true, a, X_new = None, Y_new = None):
    parities = []
    impacts = []
    eq_odds = []
    metric_frames = []
    metrics = {
        'false positive rate': false_positive_rate,
        'true positive rate': true_positive_rate
    }

    metric_frame = MetricFrame(metrics, y_true, y_pred, sensitive_features=a)
    return metric_frame.by_group["true positive rate"], metric_frame.by_group["false positive rate"]


@timeout(int(args.time_out))
def test_cases(dataset, program_name, max_iter, X_train, X_test,X_test_og, y_train, y_test, sensitive_param, group_0, group_1, sensitive_name, start_time):
    num_args = 0
    wanted_tests = pd.read_csv("tested_parfait-ml_counterfactual_summary.csv")
    wanted_tests = wanted_tests[wanted_tests["dataset"] == f"{dataset}, {sensitive_name}"]
    if(program_name == "LogisticRegression"):
        import LogisticRegression_Update
        input_program = LogisticRegression_Update.logistic_regression
        input_program_tree = 'logistic_regression_Params.xml'
        num_args = 15
        wanted_tests = wanted_tests[wanted_tests["model"] == "Logistic Regression"]
        print(wanted_tests)
    elif(program_name == "Decision_Tree_Classifier"):
        import Decision_Tree_Classifier_Update
        input_program = Decision_Tree_Classifier_Update.DecisionTree
        input_program_tree = 'Decision_Tree_Classifier_Params.xml'
        num_args = 13
        wanted_tests = wanted_tests[wanted_tests["model"] == "Decision Tree"]
        print(wanted_tests)
    elif(program_name == "TreeRegressor"):
        import TreeRegressor_Update
        input_program = TreeRegressor_Update.TreeRegress
        input_program_tree = 'TreeRegressor_Params.xml'
        num_args = 18
        wanted_tests = wanted_tests[wanted_tests["model"] == "Random Forest"]
        print(wanted_tests)
    elif(program_name == "SVM"):
        import SVM_Update
        input_program = SVM_Update.SVM
        input_program_tree = 'SVM_Params.xml'
        num_args = 15
        wanted_tests = wanted_tests[wanted_tests["model"] == "Support Vector Machine"]
        print(wanted_tests)

    arr_min, arr_max, arr_type, arr_default = xml_parser_domains.xml_parser_domains(input_program_tree, num_args)

    promising_inputs_fair1 = []
    promising_inputs_fair2 = []
    promising_inputs_AOD = []
    promising_metric_fair1 = []
    promising_metric_fair2 = []
    promising_metric_AOD = []


    high_diff_1 = 0.0
    high_diff_2 = 0.0
    low_diff_1 = 1.0
    low_diff_2 = 1.0
    default_acc = 0.0
    failed = 0
    highest_acc = 0.0
    highest_acc_inp = None
    AOD_diff = 0.0

    if args.output == None:
        filename = "./updated_Dataset/" + program_name + "_" +  dataset + "_" + sensitive_name + "_mutation_" + str(int(start_time)) + "_res.csv"
    elif args.output == "":
        filename = "./updated_Dataset/" + program_name + "_" +  dataset + "_" + sensitive_name + "_mutation_" + str(int(start_time)) + "_res.csv"
    elif ".csv" in args.output:
        filename = "./updated_Dataset/" + args.output
    else:
        filename = "./updated_Dataset/" + args.output + ".csv"

    with open(filename, 'w') as f:
        counter = 0
        

        for row_index, row in wanted_tests.iterrows():
            arr = row["write_file"].split("_")[4:]
            arr[-1] = arr[-1][:-4] # Remove the .pkl
            if arr[0] == "log" and arr[1] == "loss":
                arr[0] = "log_loss"
                arr.pop(1)
            arr = [int(a) if (isfloat(a) and ('.' not in a)) else (float(a) if isfloat(a) else (a == "True" if a == "True" or a =="False" else a)) for a in arr]
            
            # For SVM, remove duplicated arr[11]
            # For decision tree, append arr[10]
            if row["model"] == "Support Vector Machine":
                arr.pop(11)
            if row["model"] == "Decision Tree":
                arr.insert(10, 0.0)
            print(arr)
            inp = []
            # include default value
            if counter == 0:
                for i in range(len(arr_min)):
                    if(arr_type[i] == 'bool'):
                        inp.append(int(arr_default[i]))
                    elif(arr_type[i] == 'int'):
                        inp.append(int(arr_default[i]))
                    elif(arr_type[i] == 'float'):
                        inp.append(float(arr_default[i]))
            else:
                rnd = np.random.random()
                if (rnd < 0.05 and counter > 100) or (rnd < 0.5 and counter < 100):
                    for i in range(len(arr_min)):
                        if(arr_type[i] == 'bool'):
                            inp.append(randint.rvs(0,2))
                        elif(arr_type[i] == 'int'):
                            minVal = int(arr_min[i])
                            maxVal = int(arr_max[i])
                            inp.append(np.random.randint(minVal,maxVal+1))
                        elif(arr_type[i] == 'float'):
                            minVal = float(arr_min[i])
                            maxVal = float(arr_max[i])
                            inp.append(np.random.uniform(minVal,maxVal+0.00001))
                else:
                    # if rnd < 0.9:
                    inp = promising_inputs_AOD[-1]
                    print(inp)
                    index = np.random.randint(0,len(arr_min)-1)
                    if(arr_type[index] == 'bool'):
                        inp[index] = 1 - inp[index]
                    elif(arr_type[index] == 'int'):
                        minVal = int(arr_min[index])
                        maxVal = int(arr_max[index])
                        rnd = np.random.random()
                        if rnd < 0.4:
                            newVal = np.random.randint(minVal,maxVal+1)
                            trail = 0
                            while newVal == inp[index] and trail < 3:
                                newVal = np.random.randint(minVal,maxVal+1)
                                trail += 1
                        elif rnd < 0.7:
                            newVal = inp[index] + 1
                        else:
                            newVal = inp[index] - 1
                        inp[index] = newVal
                    elif(arr_type[index] == 'float'):
                        minVal = float(arr_min[index])
                        maxVal = float(arr_max[index])
                        rnd = np.random.random()
                        if rnd < 0.5:
                            inp[index] = np.random.uniform(minVal,maxVal+0.000001)
                        elif rnd < 0.75:
                            newVal = inp[index] + abs(maxVal-minVal)/100
                        else:
                            newVal = inp[index] - abs(maxVal-minVal)/100
            print(inp)
            if (args.standard_scale=="True"):
                from sklearn.preprocessing import StandardScaler
                # To avoid "data leaking"/contaminating the testing data, we transform/fit the X_test data using the X_train data. 
                ss = StandardScaler()
                ss.fit(X_train)
                
                res, LR, inp_valid, score, preds, features, write_file = input_program(inp, ss.transform(X_train), ss.transform(X_test), y_train, y_test,arr,  sensitive_param, dataset_name=dataset, save_model=(args.save_model=="True"))
            else:
                res, LR, inp_valid, score, preds, features, write_file = input_program(inp, X_train, X_test, y_train, y_test,arr, sensitive_param, dataset_name=dataset, save_model=(args.save_model=="True"))
            if not res:
                failed += 1
                continue

            if counter == 0:
                features.append("score")
                features.append("AOD")
                features.append("TPR")
                features.append("FPR")
                features.append("counter")
                features.append("timer")
                features.append("write_file")
                for i in range(len(features)):
                    if i < len(features) - 1:
                        if features[i] == None:
                            f.write(",")
                        else:
                            f.write("%s," % features[i])
                    else:
                        f.write("%s" % features[i])
                f.write("\n")
                default_acc = score

            # if (score < (default_acc - 0.01)): # TODO: Accuracy 1%
            #    continue

            if(score > highest_acc):
                highest_acc = score
                highest_acc_inp = inp_valid

            fair_metric_1, fair_metric_2 = check_for_fairness(X_test_og, preds, y_test, X_test_og[:,sensitive_param-1])

            diff_1 = np.abs(fair_metric_1[group_0] - fair_metric_1[group_1])
            diff_2 = np.abs(fair_metric_2[group_0] - fair_metric_2[group_1])

            AOD = (diff_1 + diff_2) * 0.5

            full_inp = inp_valid.copy()
            full_inp.append(score)
            full_inp.append(AOD)
            full_inp.append(diff_1)
            full_inp.append(diff_2)
            full_inp.append(counter)
            full_inp.append(time.time() - start_time)
            full_inp.append(f'"{str(write_file)}"')

            for i in range(len(full_inp)):
                if i < len(full_inp) - 1:
                    if full_inp[i] == None:
                        f.write(",")
                    else:
                        f.write("%s," % full_inp[i])
                else:
                    f.write("%s" % full_inp[i])
            f.write("\n")

            if AOD_diff < AOD:
                promising_inputs_AOD.append(inp)
                promising_metric_AOD.append([AOD, score])
                AOD_diff = AOD

            if high_diff_1 < diff_1:
                promising_inputs_fair1.append(inp)
                promising_metric_fair1.append([diff_1, score])
                high_diff_1 = diff_1

            if high_diff_2 < diff_2:
                promising_inputs_fair2.append(inp)
                promising_metric_fair2.append([diff_2, score])
                high_diff_2 = diff_2

            if low_diff_1 > diff_1:
                low_diff_1 = diff_1

            if low_diff_2 > diff_2:
                low_diff_2 = diff_2

            if counter == 0:
                promising_inputs_fair1.append(inp)
                promising_inputs_fair2.append(inp)
                promising_inputs_AOD.append(inp)
                promising_metric_fair1.append([diff_1, score])
                promising_metric_fair2.append([diff_2, score])
                promising_metric_AOD.append([AOD, score])
                high_diff_1 = diff_1
                high_diff_2 = diff_2

            print("Highest AOD difference is " + str(AOD_diff))
            print("Highest EOD different is " + str(high_diff_1))
            print("score is " + str(score))
            print("counter: " + str(counter))
            print("---------------------------------------------------------")
            counter += 1

    print("------------------END-----------------------------------")
    print(promising_inputs_fair1[-1])
    print(promising_inputs_fair1[0])
    print(promising_inputs_fair2[-1])
    print(promising_inputs_fair2[0])
    print(promising_metric_fair1[-1])
    print(promising_metric_fair1[0])
    print(promising_metric_fair2[-1])
    print(promising_metric_fair2[0])
    print("Highest AOD differences " + str(AOD_diff))
    print("Lowest fairness (1) differences " + str(low_diff_1))
    print("Lowest fairness (2) differences " + str(low_diff_2))
    print("Failed Test cases: " + str(failed))
    print("Highest accuracy observed: " + str(highest_acc))
    print("Highest accuracy input: " + str(highest_acc_inp))


def mask_relationship(X):
    X[:,6][X[:,6] == 2] = 0

def mask_age(X):
    X[:,0] = 3

if __name__ == '__main__':
    start_time = time.time()
    dataset = args.dataset
    # algorithm = LogisticRegression, Decision_Tree_Classifier, TreeRegressor, Discriminant_Analysis
    algorithm = args.algorithm
    num_iteration =  int(args.max_iter)

    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas}

    # census (9 is for sex: 0 (men) vs 1 (female); 8 is for race: 0 (white) vs 4 (black))
    # credit ...
    # bank ...
    sensitive_param = int(args.sensitive_index)
    sensitive_name = ""

    group_0 = 0
    group_1 = 1
    if dataset == "census" and sensitive_param == 9:
        sensitive_name = "gender"
        group_0 = 0  #female
        group_1 = 1  #male
    if dataset == "census" and sensitive_param == 8:
        group_0 = 0 # white
        group_1 = 4 # black
        sensitive_name = "race"
    if dataset == "credit" and sensitive_param == 9:
        group_0 = 0  # male
        group_1 = 1  # female
        sensitive_name = "gender"
    if dataset == "bank" and sensitive_param == 1:  # with 3,5: 0.89; with 2,5: 0.84; with 4,5: 0.05; with 3,4: 0.6
        group_0 = 3
        group_1 = 5
        sensitive_name = "age"
    if dataset == "compas" and sensitive_param == 1:  # sex
        group_0 = 0 # male
        group_1 = 1 # female
        sensitive_name = "gender"
    if dataset == "compas" and sensitive_param == 2:  # age
        group_0 = 0 # under 25
        group_1 = 2 # greater than 45
        sensitive_name = "age"
    if dataset == "compas" and sensitive_param == 3:  # race
        group_0 = 0 # non-Caucasian
        group_1 = 1 # Caucasian
        sensitive_name = "race"



    X, Y, input_shape, nb_classes = data[dataset]()

    if dataset == "census":
        mask_relationship(X)



    Y = np.argmax(Y, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    X_test_og = X_test.copy()
    if dataset == "bank":
        mask_age(X_train)
        mask_age(X_test)

    
    try:
        test_cases(dataset, algorithm, num_iteration, X_train, X_test,X_test_og, y_train, y_test, sensitive_param, group_0, group_1, sensitive_name, start_time)
    except TimeoutError as error:
        print("Caght an error!" + str(error))
        print("--- %s seconds ---" % (time.time() - start_time))

    print("--- %s seconds ---" % (time.time() - start_time))
