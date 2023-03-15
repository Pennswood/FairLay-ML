import numpy as np
import pandas as pd
import plotly.express as plt
import plotly.tools as tools
import math
import pickle
import csv
import os
import matplotlib.pyplot as mplt
import sys
sys.path.append("../")
from configs import columns, get_groups, labeled_df, categorical_features, categorical_features_names, int_to_cat_labels_map, cat_to_int_map
from Themis.Themis2.themis2 import Themis

### NOTE: I updated the settings under Themis/Themis2/settings_{dataset[0] (pseudocode)} to generate 1000 datapoints exactly (both min and max was set to 1000)
### This should generate 1000*6*3*(num studies) datapoints for causal, and 1000*6*3*(sum: num sensitve groups for each study) for group for each dataset.

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def dict_to_num_encoding_list(input_dict, dataset):
    output = []
    for k, v in input_dict.items():
        if columns[dataset[0]].index(k) in categorical_features[dataset[0]][:-1]:
            output.append(int(cat_to_int_map(dataset[0])[k][v]))
        else:
            output.append(int(v))
    return output

# Plots all the graphs
def create_datasets(model, dataset, algo):

    # Plot the pareto optimal frontier

    df = pd.read_csv(os.path.dirname(__file__)+"/../../Dataset" + "/" + f"{model}_{dataset[0]}_{dataset[1]}_{algo}.csv")
    df_masking = df.copy()
    df_masking["score"] = -df_masking["score"] # we want to find maximium for score
    mask = is_pareto_efficient(df_masking[["score","AOD"]].to_numpy(), True)
    df = df.assign(pareto_optimal=mask)

    # Get themis data
    df = df.assign(themis_group_score="NA")
    df = df.assign(themis_causal_score="NA")
    df_pareto_optimal = df[df["pareto_optimal"]]
    count = 0

    df_worst = df[df["AOD"] == df["AOD"].max()]
    df_pareto_optimal_max_AOD_row = df_pareto_optimal[df_pareto_optimal["AOD"] == df_pareto_optimal["AOD"].min()]
    df_pareto_optimal_max_score_row = df_pareto_optimal[df_pareto_optimal["score"] == df_pareto_optimal["score"].max()]

    for row in [df_worst, df_pareto_optimal_max_AOD_row, df_pareto_optimal_max_score_row]:

        themis_studying_feature = [dataset[1] if dataset[1] != "gender" else "sex"]
        tests = [{"function": "causal_discrimination", "threshold": 0.2, "conf": 0.98, "margin": 0.02, "input_name": themis_studying_feature},
            {"function": "group_discrimination", "threshold": 0.2, "conf": 0.98, "margin": 0.02, "input_name": themis_studying_feature}]
        write_file = row['write_file'].iloc[0]
        file_path = os.path.realpath(os.path.dirname(__file__))
        file = open(file_path+"/../"+f".{write_file}", "rb")
        trained_model = pickle.load(file, encoding="latin-1")
        S = Themis_S(trained_model, dataset)
        themis_results = Themis(S, tests, f"{file_path}/../Themis/Themis2/settings_{dataset[0]}.xml").run()


        causal_test_suite = themis_results[0][1][0]
        causal_test_suite = [dict_to_num_encoding_list(test_data, dataset) for test_data in causal_test_suite]

        group_test_suite = themis_results[1][1][0]
        group_test_suite = [dict_to_num_encoding_list(test_data, dataset) for test_data in group_test_suite]
        
        print("Causal test stuite")
        write_data(causal_test_suite, f"causal_{dataset[0]}")
        
        print("Group test stuite")
        write_data(group_test_suite, f"group_{dataset[0]}")
        
    

    return df_worst[["score", "AOD", "themis_group_score", "themis_causal_score"]], df_pareto_optimal_max_score_row[["score", "AOD", "themis_group_score", "themis_causal_score"]], df_pareto_optimal_max_AOD_row[["score", "AOD", "themis_group_score", "themis_causal_score"]]

    


def Themis_S(trained_model, dataset):
    def Themis_S(x):
        intergerized_x = []
        for i in range(len(x)):
            if i in categorical_features[dataset[0]][:-1]:
                intergerized_x.append(int(cat_to_int_map(dataset[0])[columns[dataset[0]][i]][x[i]]))
            else:
                intergerized_x.append(int(x[i]))
        return list(trained_model.predict([intergerized_x])) == [trained_model.classes_[0]]
    return Themis_S


def write_data(data, name):
    with open(f'datasets/{name}_datafile.csv', 'a') as file:
        output = csv.writer(file)
        output.writerows(data)

def main():
    models = ["LR","RF","SV","DT"]
    models = ["Logistic Regression","Random Forest","Support Vector Machine","Decision Tree"]

    models_key = {"Logistic Regression":"LR", "Random Forest": "RF", "Support Vector Machine": "SV", "Decision Tree": "DT"}
    # Note: Index starts at 1 for the datasets, so subtract 1 from datasets[2] to ensure we are highlighting the correct sensitive feature!
    datasets = [("census", "gender",9), ("census", "race",8), ("credit", "gender",9), ("bank","age",1), ("compas","gender",1), ("compas","race",3)]

    algorithms = ["mutation"]

    # Basic buttons to choose the correct models
    
    picked_algo = "mutation"
    count = 0
    score_dataframe = pd.DataFrame(columns = ['score', 'AOD', 'themis_group_score', 'themis_causal_score', 'dataset', 'model', 'optimal'])

    datasets_cols = {"bank": [["age","job","martial","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]],
                    "census": [["age","workclass","fnlwgt","education","martial-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]],
                    "credit": [["checking-account-status","duration-months","credit-history","purpose","credit-amount","saving-account-status","employment-status","installment-rate","sex","other-debts","current-residence-time","property-type","age","installment-plans","housing","num-credit-lines","job-type","num-dependents","telephone","foreigner"]],
                    "compas": [["sex","age","race","juvile-felony-count","decile-score","prior-offense-counts","g","h","i","j","k","prior-offense-counts"]]}
    for d in ["census", "credit", "bank", "compas"]:
        write_data(datasets_cols[d], f"causal_{d}")
        write_data(datasets_cols[d], f"group_{d}")
    for d in datasets:
        for m in models:
            create_datasets(models_key[m], d, picked_algo)


    # convert_df(score_dataframe)

    
if __name__ == "__main__":
    main()