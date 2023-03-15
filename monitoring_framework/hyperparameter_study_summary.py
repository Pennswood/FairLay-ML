import numpy as np
import pandas as pd
import plotly.express as plt
import plotly.tools as tools
import math
import pickle
from Themis.Themis2.themis2 import Themis
import os
import matplotlib.pyplot as mplt
from configs import columns, get_groups, labeled_df, categorical_features, categorical_features_names, int_to_cat_labels_map, cat_to_int_map

import sys

sys.path.append("./")
sys.path.append("../")





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



# Plots all the graphs
def create_model(model, dataset, algo):

    # Plot the pareto optimal frontier

    df = pd.read_csv(os.path.dirname(__file__)+"/../Dataset" + "/" + f"{model}_{dataset[0]}_{dataset[1]}_{algo}.csv")
    df_masking = df.copy()
    df_masking["score"] = -df_masking["score"] # we want to find maximium for score
    mask = is_pareto_efficient(df_masking[["score","AOD"]].to_numpy(), True)
    df = df.assign(pareto_optimal=mask)

    # Get themis data
    df = df.assign(themis_group_score="NA")
    df = df.assign(themis_causal_score="NA")
    df_pareto_optimal = df[df["pareto_optimal"]]
    # count = 0

    df_worst = df[df["AOD"] == df["AOD"].max()]
    df_pareto_optimal_max_AOD_row = df_pareto_optimal[df_pareto_optimal["AOD"] == df_pareto_optimal["AOD"].min()]
    df_pareto_optimal_max_score_row = df_pareto_optimal[df_pareto_optimal["score"] == df_pareto_optimal["score"].max()]

    hyperparameters = [[]]*3 #(worst, optimal score, optimal AOD)
    i = 0
    for row in [df_worst,df_pareto_optimal_max_score_row, df_pareto_optimal_max_AOD_row]:
        write_file = row['write_file'].iloc[0]
        parameters = write_file.split("/")[-1]
        
        parameters = parameters.split("_")[3:]

        if parameters[0] == "log" and parameters[1] == "loss":
            parameters[0] = "log_loss"
            parameters.pop(1)
        
        parameters[-1] = parameters[-1][:-4] #Remove ".csv"
        if model == "LR":
            hyperparameters[i] = [parameters[1], parameters[2], parameters[3], parameters[4],parameters[5], parameters[6],parameters[0], parameters[7], parameters[8], parameters[9], parameters[11], parameters[10], parameters[12], parameters[13], parameters[14]]
        elif model == "SV":
            # I have a screw up here during my parameters save. I save parameters[11] twice, add one to index after 11
            hyperparameters[i] = [parameters[0], parameters[1], parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10],parameters[11],parameters[13],parameters[14],parameters[15]] 
        elif model == "DT":
            # parameter 10 is not saved, subtract 1 from index after 10
            hyperparameters[i] = [parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11]]
        elif model == "RF":
            hyperparameters[i] = [parameters[11],parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16]]
        else:
            raise Exception("Whoops")
        i+=1
        

    return hyperparameters

    




def convert_df(df, name):
   return df.to_csv(f"{name}_hyperparameters_summary.csv", index=False)

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
    lr_dataframe = pd.DataFrame(columns = ["dataset", 'optimal', "penalty","dual", 'tol', 'C', 'fit_intercept', "intercept_scaling",
    "solver", "max_iter", "multi_class", "l1_ration", "random_state", "class_weight", "verbose", "warm_start", "n_jobs"])

    sv_dataframe = pd.DataFrame(columns = ["dataset", 'optimal', "C","kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size",
    "class_weight", "verbose", "max_iter", "decision_function_shape", "break_ties", "random_state"])

    dt_dataframe = pd.DataFrame(columns = ["dataset", 'optimal', "criterion", "splitter", "max_depth", "min_samples_split", "min_samples_leaf", 
    "min_weight_fraction_leaf", "max_features", "random_state", "max_leaf_nodes", "min_impurity_decrease", "class_weight", "ccp_alpha"])

    rf_dataframe = pd.DataFrame(columns = ["dataset", 'optimal', "n_estimators", "criterion", "max_depth", "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf",
    "max_features", "max_leaf_nodes", "min_impurity_decrease", "bootstrap", "oob_score", "warm_start", "ccp_alpha", "max_samples", "random_state", "verbose" ,"n_jobs"])



    for d in datasets:
        for m in models:
            hyperparameters = create_model(models_key[m], d, picked_algo)
            hyperparameters[0].insert(0, "Worst")
            hyperparameters[1].insert(0, "Score")
            hyperparameters[2].insert(0, "Fairness")
            for h in hyperparameters:
                h.insert(0, d[0] + ", " + d[1])
                if models_key[m] == "LR":
                    lr_dataframe.loc[len(lr_dataframe.index)] = h
                elif models_key[m] == "SV":
                    sv_dataframe.loc[len(sv_dataframe.index)] = h
                elif models_key[m] == "DT":
                    dt_dataframe.loc[len(dt_dataframe.index)] = h
                elif models_key[m] == "RF":
                    rf_dataframe.loc[len(rf_dataframe.index)] = h
                else:
                    raise Exception("Whoops")

            

            count +=1

    convert_df(lr_dataframe, "lr")
    convert_df(sv_dataframe, "sv")
    convert_df(dt_dataframe, "dt")
    convert_df(rf_dataframe, "rf")

    
if __name__ == "__main__":
    main()