import streamlit as st
import random
import pickle
import pandas as pd

st.title("Audiogram Input Page")


# Frequencies and derived keys
frequencies = ["125", "250", "500", "750", "1000", "1500", "2000", "3000", "4000", "6000", "8000"]
left_keys = [f"hz{f}_L" for f in frequencies]
right_keys = [f"hz{f}_R" for f in frequencies]
all_keys = left_keys + right_keys + ['WRS_L', 'WRS_R', 'age']

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
    st.session_state['age'] = random.randint(18, 95)


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
st.number_input("Age", min_value=0, max_value=120, step=1, key="age")

# Prediction
if st.button("Predict Risk Score (CNC)"):
    with open('best_grid_search_10_bins_smote.pkl', 'rb') as f:
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

    # Add WRS and age
    X_left["WRS"] = [st.session_state["WRS_L"]]
    X_right["WRS"] = [st.session_state["WRS_R"]]
    X_left["age"] = [st.session_state["age"]]
    X_right["age"] = [st.session_state["age"]]

    # Convert to DataFrames
    X_left_df = pd.DataFrame(X_left)
    X_right_df = pd.DataFrame(X_right)

    # Predict
    risk_pred_L = grid_search.best_estimator_.predict(X_left_df)[0]  #The percent change that CNC will be ABOVE 50
    risk_pred_R = grid_search.best_estimator_.predict(X_right_df)[0]

    below_50_prob_L = (10-risk_pred_L)*10
    below_50_prob_R = (10-risk_pred_R)*10

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

