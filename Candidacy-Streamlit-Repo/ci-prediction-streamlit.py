import streamlit as st
import random
import pickle
import pandas as pd
import sklearn
from xgboost import XGBClassifier

###Purely just a UI that loads in the ML model specified and uses it to generate the "Calculator" and corresponding risk score



st.title("Audiogram Input Page")


# Frequencies and derived keys
frequencies = ["125", "250", "500", "750", "1000", "1500", "2000", "3000", "4000", "6000", "8000"]
left_keys = [f"hz{f}_L" for f in frequencies]
right_keys = [f"hz{f}_R" for f in frequencies]
all_keys = left_keys + right_keys + ['WRS_L', 'WRS_R', 'Age']

# Initialize session state
for key in all_keys:
    if key not in st.session_state:
        st.session_state[key] = 0

# Random value generator
if st.button("Generate Random Audiogram Data"):
    for key in left_keys + right_keys:
        st.session_state[key] = random.randint(0, 120)
    st.session_state['WRS_L'] = random.randint(0, 100)
    st.session_state['WRS_R'] = random.randint(0, 100)
    st.session_state['Age'] = random.randint(18, 95)


# Input layout function
def input_table(title, keys, side_label, columns_per_row=4):
    st.subheader(title)
    for i in range(0, len(keys), columns_per_row):
        row_keys = keys[i:i+columns_per_row]
        cols = st.columns(len(row_keys))
        for col, key in zip(cols, row_keys):
            freq = key.split("_")[0][2:]
            with col:
                st.number_input(
                    f"{freq} Hz\n({side_label})",
                    min_value=0,
                    max_value=120,
                    step=1,
                    key=key,
                    label_visibility="visible"
                )

# Audiogram inputs
input_table("Left Audiogram Data", left_keys, "Left", columns_per_row=4)
st.number_input("WRS (Left)", min_value=0, max_value=100, step=1, key="WRS_L")
input_table("Right Audiogram Data", right_keys, "Right", columns_per_row=4)
st.number_input("WRS (Right)", min_value=0, max_value=100, step=1, key="WRS_R")

# Age input (shared)
st.number_input("Age", min_value=0, max_value=120, step=1, key="Age")

# Prediction for CNC
if st.button("Predict Risk Score (CNC)"):
    with open('Candidacy-Streamlit-Repo/best_grid_search_10_bins_percentiles.pkl', 'rb') as f:
        grid_search = pickle.load(f)

    # Build cleaned feature sets
    X_left = {
        key.replace("_L", ""): [st.session_state[key]]
        for key in left_keys
    }
    X_right = {
        key.replace("_R", ""): [st.session_state[key]]
        for key in right_keys
    }

    # Add WRS and Age
    X_left["WRS"] = [st.session_state["WRS_L"]]
    X_right["WRS"] = [st.session_state["WRS_R"]]
    X_left["Age"] = [st.session_state["Age"]]
    X_right["Age"] = [st.session_state["Age"]]

    # Convert to DataFrames
    X_left_df = pd.DataFrame(X_left)
    X_right_df = pd.DataFrame(X_right)


    ##########CNC Risk Prediction#############
    # Predict
    risk_pred_L = grid_search.best_estimator_.predict_proba(X_left_df)  #The percent change that CNC will be ABOVE 50
    risk_pred_R = grid_search.best_estimator_.predict_proba(X_right_df)

    print(risk_pred_L)
    # Classes to sum for binary prob
    pred_prob_below_thresh_L = risk_pred_L[:, :6].sum(axis=1)
    pred_prob_below_thresh_R = risk_pred_L[:, :6].sum(axis=1)

    below_50_prob_L = int(pred_prob_below_thresh_L* 100)
    below_50_prob_R = int(pred_prob_below_thresh_R* 100)

    # Display
    def display_colored_risk(label, value):
        if value < 40:
            color = "green"
        elif value < 80:
            color = "orange"
        else:
            color = "red"

        st.markdown(
            f"<div style='font-size:18px; font-weight:bold; color:{color};'>"
            f"{label}: {value:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )


    # Usage
    display_colored_risk("Predicted Risk of CNC Less Than 50 (Left)", below_50_prob_L)
    display_colored_risk("Predicted Risk of CNC Less Than 50 (Right)", below_50_prob_R)


if st.button("Predict Risk Score (AzBio, bilateral)"):
    with open('Candidacy-Streamlit-Repo/grid_search_azbio_bi_10.pkl', 'rb') as f:
        az_grid_search = pickle.load(f)

    # Build cleaned feature sets
    az_X_L = {
        key: [st.session_state[key]]
        for key in left_keys
    }

    az_X_R = {
        key: [st.session_state[key]]
        for key in right_keys
    }

    # Stagger/combine them
    combined = {}
    for k1, k2 in zip(right_keys, left_keys):
        combined[k1] = az_X_R[k1]
        combined[k2] = az_X_L[k2]


    az_X_combined = pd.DataFrame(combined)

    # Add WRS and age without renaming
    az_X_combined["WRS_L"] = [st.session_state["WRS_L"]]
    az_X_combined["WRS_R"] = [st.session_state["WRS_R"]]
    az_X_combined["Age"] = [st.session_state["Age"]]

    # Convert to DataFrame
    az_X_df = pd.DataFrame(az_X_combined)

    # Add WRS and Age
    az_X_df["WRS_L"] = [st.session_state["WRS_L"]]
    az_X_df["WRS_R"] = [st.session_state["WRS_R"]]
    az_X_df["Age"] = [st.session_state["Age"]]

    print(az_X_df.columns)

    ##########AzBio Risk Prediction#############
    # Predict
    az_bin_threshold = 4
    az_risk_pred = az_grid_search.best_estimator_.predict_proba(az_X_df)  # The percent change that CNC will be ABOVE 50
    # Classes to sum for binary prob
    pred_prob_below_thresh = az_risk_pred[:, :4].sum(axis=1)
    below_60_prob = int(pred_prob_below_thresh* 100)


    # Display
    def display_az_colored_risk(label, value):
        if value < 40:
            color = "green"
        elif value < 80:
            color = "orange"
        else:
            color = "red"

        st.markdown(
            f"<div style='font-size:18px; font-weight:bold; color:{color};'>"
            f"{label}: {value:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )


    # Usage
    display_az_colored_risk("Predicted Risk of AzBio Less Than 60 (Bilateral)", below_60_prob)
