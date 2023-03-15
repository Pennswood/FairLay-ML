#!/bin/sh
python3 main_mutation_custom_masking.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --time_out=$1 --output=LR_census_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation_custom_masking.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --time_out=$1 --output=RF_census_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation_custom_masking.py --dataset=census --algorithm=SVM --sensitive_index=9 --time_out=$1 --output=SV_census_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation_custom_masking.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --time_out=$1 --output=DT_census_gender_mutation --save_model=$2 --max_iter=$3

python3 main_mutation_custom_masking.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --time_out=$1 --output=LR_bank_age_mutation --save_model=$2 --max_iter=$3
python3 main_mutation_custom_masking.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --time_out=$1 --output=RF_bank_age_mutation --save_model=$2 --max_iter=$3
python3 main_mutation_custom_masking.py --dataset=bank --algorithm=SVM --sensitive_index=1 --time_out=$1 --output=SV_bank_age_mutation --save_model=$2 --max_iter=$3
python3 main_mutation_custom_masking.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --time_out=$1 --output=DT_bank_age_mutation --save_model=$2 --max_iter=$3
