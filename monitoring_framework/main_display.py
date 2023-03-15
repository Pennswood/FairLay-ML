import data_monitor
import themis_study
import study_counterfactuals
import updated_models_study
import home
import streamlit as st
import sys

sys.path.append("./")
sys.path.append("../")
st.write("NOTE: clicking on some of the plotly graphs produces more details.")
version = st.radio("Visualization tool:", ["Home", "Data Visualization", "Learned Model Explainations and Counterfactuals", "Themis Study", "Updated models counterfactuals"], horizontal = True)

if version == "Home":
    home.main()
if version == "Data Visualization":
    data_monitor.main()
elif version == "Learned Model Explainations and Counterfactuals":
    study_counterfactuals.main()
elif version == "Themis Study":
    themis_study.main()
elif version == "Updated models counterfactuals":
    updated_models_study.main()