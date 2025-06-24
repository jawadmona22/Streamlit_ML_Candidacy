import pandas as pd
from sklearn.model_selection import GroupKFold,GridSearchCV,GroupShuffleSplit,ParameterGrid,KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from sklearn.metrics import  ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
import seaborn as sns
from pycaret.classification import setup, compare_models, pull
from pycaret.classification import *
from pathlib import Path
import joblib
from scipy.stats import t
from sklearn import preprocessing


def Clean_Data(debug=False,all_labels=[]): #This function extracts the columns of interest, removes NA, and adds patient ID
    full_dataset = pd.read_csv("candidacy_v3.csv")
    print("Cleaning data...")
    if 'Age' in full_dataset.columns:
        # Calculate median age (ignores NaNs by default)
        median_age = full_dataset["Age"].median()
        # print("Missing before:", full_dataset["Age"].isna().sum())
        full_dataset["Age"].fillna(median_age, inplace=True)
        # print("Missing after:", full_dataset["Age"].isna().sum())

    if 'HLdur_R' and 'HLdur_L' in full_dataset.columns:
        # First for right
        # Calculate median age (ignores NaNs by default)
        median_HL_r = full_dataset["HLdur_R"].median()
        full_dataset["HLdur_R"].fillna(median_HL_r, inplace=True)

        # Then for left
        median_HL_l = full_dataset["HLdur_L"].median()
        full_dataset["HLdur_L"].fillna(median_HL_l, inplace=True)

    if {'Hearing_Aid_Use_Time_R', 'Hearing_Aid_Use_Time_L', 'HearingAidUse'}.issubset(full_dataset.columns):
        # Set to 0 where HearingAidUse is "No"
        full_dataset.loc[full_dataset['HearingAidUse'] == "No",
        ['Hearing_Aid_Use_Time_R', 'Hearing_Aid_Use_Time_L']] = 0

        # Fill missing HA Use Time R with median for HearingAidUse != "No"
        mask_r = (full_dataset['HearingAidUse'] != "No") & (full_dataset['Hearing_Aid_Use_Time_R'].isna())
        median_ha_r = full_dataset.loc[full_dataset['HearingAidUse'] != "No", "Hearing_Aid_Use_Time_R"].median()
        full_dataset.loc[mask_r, "Hearing_Aid_Use_Time_R"] = median_ha_r

        # Fill missing HA Use Time L with median for HearingAidUse != "No"
        mask_l = (full_dataset['HearingAidUse'] != "No") & (full_dataset['Hearing_Aid_Use_Time_L'].isna())
        median_ha_l = full_dataset.loc[full_dataset['HearingAidUse'] != "No", "Hearing_Aid_Use_Time_L"].median()
        full_dataset.loc[mask_l, "Hearing_Aid_Use_Time_L"] = median_ha_l

    #Need to keep everything together to prevent loss of data
    filtered_dataset = full_dataset[all_labels].dropna()
    filtered_dataset['patient_id'] = range(1, len(filtered_dataset) + 1)
    # if debug:
        # print(f"Filtered Dataset: {filtered_dataset.head}")
        # print(f"Filtered Columns: {filtered_dataset.columns}")


    return filtered_dataset






def Create_Left_Right_Data(unseparated_data,debug=False):
#Data from L/R is combined so that they are unlabeled, but patient IDs are preserved'''

    df = pd.DataFrame(unseparated_data)
    # Separate L and R columns
    left_df = df.filter(regex='_L$').copy()
    right_df = df.filter(regex='_R$').copy()

    # Extract the patient_id column
    patient_ids = df['patient_id'].copy()
    if 'Age' in df.columns:
        ages = df['Age'].copy()

# Rename columns by removing the side-specific suffix
    left_df.columns = left_df.columns.str.replace('_L$', '', regex=True)
    right_df.columns = right_df.columns.str.replace('_R$', '', regex=True)

    # Concatenate the dataframes
    left_df['patient_id'] = patient_ids
    right_df['patient_id'] = patient_ids

    #Add in age as well
    if 'Age' in df.columns:
        left_df['Age'] = ages
        right_df['Age'] = ages
    left_right_data = pd.concat([left_df, right_df], ignore_index=True)

    if debug:
        print(f"Data split into l/r data columns: {left_right_data.head}")
        if left_right_data.shape[0] != (df.shape[0] * 2):
            #We should be doubling the patient data
            print("WARNING! Shape is not correct for L/R Data")
        patient1_data = left_right_data[left_right_data['patient_id']==1]
        print(patient1_data)
    return left_right_data

def Add_Categorical_Bins(left_right_data,num_bins,debug=True):
    #E.G The 25th percentile means 25% of the data is less than or equal to that value.
    percentiles = np.linspace(0,100,num_bins+2)
    binned_data = left_right_data.copy()
    thresholds = np.unique(np.percentile(left_right_data['CNC'],percentiles))
    binned_data['CNC_bin'] = np.digitize(left_right_data['CNC'], thresholds, right=True)
    # Identify bin threshold below 50
    fifty_threshold = max(i for i, upper in enumerate(thresholds) if upper < 50)
    if debug:
        plt.figure(figsize=(10, 6))
        plt.hist(binned_data['CNC'], bins=thresholds, edgecolor='black', alpha=0.7)
        plt.axvline(x=thresholds[fifty_threshold], color='red', linestyle='--', label='CNC 50 Cutoff')
        plt.legend()
        plt.plot()
        plt.title('Distribution of CNC Bins (Percentile-based)')
        plt.xlabel('CNC Score')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        print(f"Percentiles: {percentiles}")
        print(f"Thresholds {thresholds}")
        print(f"Threshold for CNC 50 is bin {fifty_threshold}")
        print(f"Data Bins:{binned_data['CNC_bin'] }")
        print(f"CNC for Bin 1: {binned_data[binned_data['CNC_bin']==1]['CNC']}")
    return binned_data, fifty_threshold


def Train_Test_Split(binned_data,debug=False,raw=False):
    if raw == False:
        X = binned_data.drop(columns=['CNC_bin', 'CNC', 'patient_id'])
        y = binned_data['CNC_bin']
        groups = binned_data['patient_id'].values
        # Train-test split
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups[train_idx]  # Extract corresponding train groups

        if debug:
            print(f"Columns in X_Train: {X_train.columns}")
            print(f"Columns in X_Test: {X_test.columns}")

        return X_train,X_test,y_train,y_test, groups_train
    else:
        X = binned_data.drop(columns=['CNC', 'patient_id'])
        y = binned_data['CNC']
        groups = binned_data['patient_id'].values
        # Train-test split
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups[train_idx]  # Extract corresponding train groups

        if debug:
            print(f"Columns in X_Train: {X_train.columns}")
            print(f"Columns in X_Test: {X_test.columns}")

        return X_train, X_test, y_train, y_test, groups_train


def generate_repeated_group_kfold(X, y, groups, n_splits=10, n_repeats=100, random_state=42):

    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)
    all_splits = []

    for repeat in range(n_repeats):
        # Shuffle the unique group labels
        shuffled_groups = rng.permutation(unique_groups)

        # Map each group to a new shuffled index
        group_to_fold_index = {group: i for i, group in enumerate(shuffled_groups)}

        # Apply that mapping to the full group array to change fold assignments
        shuffled_group_labels = np.array([group_to_fold_index[group] for group in groups])

        # Create a new GroupKFold and generate splits using the shuffled labels
        group_kfold = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in group_kfold.split(X, y, groups=shuffled_group_labels):
            all_splits.append((train_idx, test_idx))

    return all_splits

def Optimize_Model_Repeated_Iters(X,y,groups,debug=False,raw=False,soft_label=False,pkl_name = 'grid_search'):
    # --- Classifier-specific hyperparameter grids ---
    cv = generate_repeated_group_kfold(X,y,groups,n_splits=10,n_repeats=100,random_state=42)
    model_configs = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, None]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
            'param_grid': {
                'penalty': ['l2']
            }
        }
    }

    # --- Store all results ---
    all_predictions = []
    # --- Loop over each classifier type ---
    for model_name, config in tqdm(model_configs.items(), desc="Models", position=0):
        base_model = config['model']
        param_grid = config['param_grid']
        model_predictions = []

        print(f"\n🔍 Running grid search for {model_name}...")

        for params in tqdm(ParameterGrid(param_grid), desc=f"{model_name} Grid", position=1, leave=False):
            # print(f"  ➤ Params: {params}")
            model = base_model.set_params(**params)
            #K-fold loop
            for fold_idx, (train_idx, test_idx) in tqdm(
                    list(enumerate(cv)),
                    total=len(cv),
                    desc="Folds",
                    position=2,
                    leave=False
            ):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                group_test = groups[test_idx]

                # Train and predict
                model.fit(X_train, y_train)
                y_full = pd.concat([y_test,y_train],axis=0,ignore_index=True)
                X_full = pd.concat([X_test, X_train], axis=0, ignore_index=True)

                y_pred = model.predict(X_full)

                # Predict probabilities
                y_pred_proba = model.predict_proba(X_full)

                pred_df = pd.DataFrame({
                    'y_true': y_full.values,
                    'y_pred': y_pred,
                    # 'patient_id': group_test,
                    'fold': fold_idx,
                    'model': model_name,
                    'params': [str(params)] * len(y_full),
                })

                proba_column = [row for row in y_pred_proba]  # Each row is a list of class probs

                # Add to prediction DataFrame
                pred_df["proba_vector"] = proba_column


                all_predictions.append(pred_df)
                model_predictions.append(pred_df)
            intermediate_df = pd.concat(model_predictions, ignore_index=True)
            intermediate_df.to_pickle(f"{model_name}_iters_{pkl_name}.pkl")

            all_predictions.extend(model_predictions)
            # --- Combine everything into one DataFrame ---
    full_predictions = pd.concat(all_predictions, ignore_index=True)
    full_predictions.to_pickle(f"All_Iter_Predictions_{pkl_name}.pkl")
def Optimize_Model(X_train,y_train,groups_train,debug=False,raw=False,soft_label=False,pkl_name = 'grid_search'):
    '''This section will take the selected model in 'params' and train the model'''

    # kf = GroupKFold(n_splits=10)
    print(f"Shape of X: {X_train.shape}")
    cv_splits = generate_repeated_group_kfold(X_train, y_train, groups=groups_train, n_splits=2, n_repeats=2)

    if raw == False: #Case: classification problem
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))  # Placeholder, will be overridden
        ])
        # Define parameter grids for different classifiers
        param_grid = [
            {  # Random Forest
                'classifier': [RandomForestClassifier(random_state=42)],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            {  # XGBoost
                'classifier': [XGBClassifier( eval_metric='logloss', random_state=42)],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 6, 10],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            },
            {  # Vanilla Logistic Regression (no hyperparameter tuning)
                'classifier': [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)]
                # No hyperparameters specified
            }
        ]

        # Perform GridSearchCV with GroupKFold
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring='f1_micro',
            cv=cv_splits,
            n_jobs=-1,
            verbose=3
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        with open(f'{pkl_name}.pkl', 'wb') as f:
            pickle.dump(grid_search, f)

        if debug:

            y_pred = grid_search.best_estimator_.predict(X_train)
            # Extract the classifier name from the pipeline
            model_name = grid_search.best_estimator_.named_steps['classifier'].__class__.__name__

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(y_train, y_pred, ax=ax)
            ax.set_title(f"Training Data Confusion Matrix {model_name}")
            plt.show()
        return grid_search



def bootstrap_roc_auc(y_true, y_score, n_bootstraps=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        # if len(np.unique(y_true[indices])) < 2:
        #     continue  # skip if only one class present
        fpr, tpr, _ = roc_curve(y_true[indices], y_score[indices])
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    tprs_lower = np.clip(mean_tpr - 1.96 * std_tpr, 0, 1)
    print(tprs_lower)
    tprs_upper = np.clip(mean_tpr + 1.96 * std_tpr, 0, 1)

    return base_fpr, mean_tpr, tprs_lower, tprs_upper
def run_visualizations(grid_search,X_train,y_train,X_test,y_test,bin_threshold,raw=False,label='Default',full=False):
    best_models = {}
    model_performance = []
    if full == True:
        X_test = pd.concat([X_test, X_train], axis=0, ignore_index=True)
        y_test = pd.concat([y_test,y_train],axis=0,ignore_index=True)

    for i, param in enumerate(grid_search.cv_results_['params']):
        model_name = param['classifier'].__class__.__name__
        mean_score = grid_search.cv_results_['mean_test_score'][i]

        # Only store the best param set per model
        if model_name not in best_models or mean_score > best_models[model_name]['score']:
            best_models[model_name] = {
                'params': param,
                'score': mean_score
            }

    # Refit the best model for each model type
    for model_name in best_models:
        best_param = best_models[model_name]['params']
        model_pipeline = grid_search.estimator.set_params(**best_param)
        model_pipeline.fit(X_train, y_train)
        classifier = model_pipeline.named_steps['classifier']
        best_models[model_name]['model'] = classifier  # Store fitted classifier
        plt.close('all')
        plt.figure()
    for model_name, model_info in best_models.items():
        best_model = model_info['model']
        print(f"Running model: {model_name}")
        y_true_binary = (y_test <= bin_threshold).astype(int)
        y_pred_prob = best_model.predict_proba(X_test)  #Outputs an array of probabilities for each test data corresponding to the ten categories
        # Classes to sum for binary prob
        pred_prob_below_thresh = y_pred_prob[:, :bin_threshold].sum(axis=1)
        y_pred_prob_binary = (pred_prob_below_thresh >= 0.5).astype(int) #We classify as "1" if your probability is >=.5
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob_binary)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC AUC Curve (Threshold = {bin_threshold})")
        plt.legend()

        # Bootstrap ROC and get confidence bounds
        fpr,tpr, tpr_lower, tpr_upper = bootstrap_roc_auc(
            y_true_binary, y_pred_prob_binary
        )
        # Compute AUC for the lower bound of the TPR
        auc_lower_bound = auc(fpr, tpr_lower)

        # Compute AUC for the upper bound of the TPR
        auc_upper_bound = auc(fpr, tpr_upper)


        tn,fp,fn,tp = confusion_matrix(y_true_binary,y_pred_prob_binary).ravel() #Ravel flattens this into a 1d Array so we can assign the variables
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # same as recall
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = precision_score(y_true_binary, y_pred_prob_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_prob_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_prob_binary, zero_division=0)


        # roc_auc = auc(fpr, mean_tpr)
        # plt.plot(fpr, mean_tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        plt.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.2)
        model_performance.append({
            'Model': model_name,
            'AUC': roc_auc,
            'CI Lower Bound': auc_lower_bound,
            'CI Upper Bound': auc_upper_bound,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    # Final plot settings
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve with 95% CI")
    plt.legend()
    plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{label}-AUC.png")
    # plt.show()

    performance_df = pd.DataFrame(model_performance)
    # Display the table of AUC and CI for each model
    performance_df.to_excel(f"performance-{label}.xlsx")

    ##Calibration curves
    # plt.figure(figsize=(8, 6))

    # Set up subplot grid
    num_models = len(best_models)
    ncols = 2
    nrows = (num_models + 1) // ncols  # Round up for uneven count
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()  # Flatten to index easily

    for idx, (model_name, model_info) in enumerate(best_models.items()):
        ax = axes[idx]
        best_model = model_info['model']
        print(f"Running model: {model_name}")

        y_true_binary = (y_test <= bin_threshold).astype(int)
        y_pred_prob = best_model.predict_proba(X_test)
        pred_prob_below_thresh = y_pred_prob[:, :bin_threshold].sum(axis=1)
        print("pred proba:",pred_prob_below_thresh)
        for i in range(len(pred_prob_below_thresh)):
            if pred_prob_below_thresh[i] >= 1:
                pred_prob_below_thresh[i] = 1

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, pred_prob_below_thresh, n_bins=10
        )

        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Calibration Curve')

        # Plot histogram for RandomForest
        # if "Random" in model_name:
        counts, bins = np.histogram(pred_prob_below_thresh, bins=10, range=(0, 1))
        proportions = counts / counts.sum()
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]
        bars = ax.bar(bin_centers, proportions, width=bin_width, alpha=0.5, label='Pred Prob Histogram')
        ax.set_ylim(0, 1)
        # Add percent text on top of each bar
        for i, bar in enumerate(bars):
            percent = proportions[i] * 100
            if percent > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,  # position just above bar
                        f"{percent:.1f}%",  # 1 decimal place
                        ha='center', va='bottom', fontsize=10)
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        ax.set_title(f"{model_name} Calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        # ax.legend(loc='best')
        ax.grid(True)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{label}-cal.png")
    # plt.show()
    #######Confusion Matrix##########
    # Get confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    #
    # # Convert to percentage
    # cm_percent = cm / cm.sum() * 100  # Normalize to sum = 100
    #
    # # Set class labels
    # labels = ['Negative', 'Positive']
    #
    # # Plot
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
    #             xticklabels=labels, yticklabels=labels)
    #
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix (% of total samples)')
    # plt.show()

def sixty_sixty_predictions(X_test):
    y_pred = []

    for index, row in X_test.iterrows():
        hz_500 = row["hz500"]
        hz_1000 = row["hz1000"]
        hz_2000 = row["hz2000"]
        wrs = row["WRS"]

        if hz_500 >= 60 and hz_1000 >= 60 and hz_2000 >= 60 and wrs < 60:
            y_pred.append(1) #They are a candidate
        else:
            y_pred.append(0)

    return y_pred


def soft_label(y, center=40, sharpness=15):
    """Smoothly map scores to probabilities that y <= center."""
    return 1 / (1 + np.exp((y - center) / sharpness))


def different_variables_run(full=False):
    ##Run 1

    set_dict = {
        # "ag-only-iters":
        # #Audiogram Only
        # [
        # 'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        # 'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        # 'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        # 'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        # 'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L', 'CNC_L', 'CNC_R'
        # ],
        # "wrs":
        # #audiogram + WRS
        # [
        # 'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        # 'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        # 'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        # 'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        # 'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        # 'WRS_L', 'WRS_R', 'CNC_L', 'CNC_R'
        # ],
        "wrs-a-iters":

        #Audiogram + WRS + Age
        [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age','CNC_L', 'CNC_R'
        ],
        # "wrs-a-dhl":
        #
        # # Audiogram + WRS + Age + HL Duration
        #     [
        #         'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        #         'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        #         'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        #         'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        #         'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        #         'WRS_L', 'WRS_R', 'Age', 'HLdur_L', 'HLdur_R','CNC_L', 'CNC_R'
        #     ],
        # "wrs-a-dhl-dha":
        # # Audiogram + WRS + Age + HL Duration + HA duration
        #     [
        #         'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        #         'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        #         'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        #         'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        #         'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        #         'WRS_L', 'WRS_R', 'Age', 'HLdur_L', 'HLdur_R', 'Hearing_Aid_Use_Time_L',
        #         'Hearing_Aid_Use_Time_R','CNC_L', 'CNC_R'
        #     ]

    }


    if not full:
        for key, value in tqdm(set_dict.items(), desc="Processing sets"):
            label = key
            all_labels = value
            main_ml_call(
                num_bins=10,
                smote=False,
                method="ML",
                raw=False,
                is_soft_label=False,
                all_labels=all_labels,
                label=label
            )

    if full:
        for key, value in tqdm(set_dict.items(), desc="Processing sets"):
            label = key
            all_labels = value
            visuals_with_full_set(all_labels,label)


def sixty_sixty_run():
    all_labels = [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
    ]

    main_ml_call(method="60/60",label="sixty",all_labels=all_labels)



def visuals_with_full_set(all_labels,label):
    from pathlib import Path
    import joblib

    folder = Path('feature-pkls')

    for pkl_file in folder.glob('*.pkl'):
        print(pkl_file.name[:-4])
        if pkl_file.name[:-4] == label:
            print(f"Loading: {pkl_file}")
            grid_search = joblib.load(pkl_file)
            filtered_dataset = Clean_Data(debug=False, all_labels=all_labels)
            left_right_data = Create_Left_Right_Data(filtered_dataset, debug=False)
            binned_data, fifty_threshold = Add_Categorical_Bins(left_right_data, num_bins=10, debug=False)
            # Split into train/test while preserving which patient is in each group
            X_train, X_test, y_train, y_test, groups_train = Train_Test_Split(binned_data, debug=False, raw=False)
            label = pkl_file.name + "-full"
            run_visualizations(grid_search, X_train, y_train, X_test, y_test, 7, raw=False, label=label,
                               full=True)

def main_ml_call(num_bins = 10, smote = False, method = "ML",raw=False,is_soft_label = False,all_labels = [],label = 'default'):

    #############################
    #Extract columns of interest, remove NA, and add patient ID and CNC bin
    filtered_dataset = Clean_Data(debug=False,all_labels=all_labels)
    left_right_data = Create_Left_Right_Data(filtered_dataset,debug=False)
    if raw == True:
        X_train, X_test, y_train,y_test,groups_train = Train_Test_Split(left_right_data, debug=False,raw=True)
        if method == "ML":
            #Run grid search on the split data **for regressor**

            if is_soft_label:
                y_train_soft = soft_label(y_train)
                x = np.linspace(0,1,len(y_train_soft))
                plt.scatter(x,y_train_soft)
                plt.show()
                y_test_soft = soft_label(y_test)

                # # Hard binary label for evaluation only
                # y_train_binary = (y_train <= 40).astype(int)
                # y_test_binary = (y_test <= 40).astype(int)
                grid_search = Optimize_Model(X_train, y_train_soft, groups_train, debug=False, raw=True,soft_label=is_soft_label,pkl_name=label)

            else:
                # grid_search = Optimize_Model(X_train, y_train, groups_train, debug=True, raw=True)

                with open(f'{label}.pkl', 'rb') as f:
                    grid_search = pickle.load(f)

                #Make y_train and y_test binary now
                y_train_binary = (y_train <=40).astype(int) #A series of 0/1
                y_test_binary = (y_test <=40).astype(int)

                best_model = grid_search.best_estimator_
                print(best_model)
                y_pred = best_model.predict(X_train)





                # run_visualizations(grid_search, X_test, y_test,bin_threshold=fifty_threshold)

            #
            # y_pred = grid_search.best_estimator_.predict(X_test)
            # model_name = grid_search.best_estimator_.named_steps['classifier'].__class__.__name__



    if raw == False:
        binned_data,fifty_threshold = Add_Categorical_Bins(left_right_data,num_bins=10,debug=False)
        print(binned_data.head)
    # Split into train/test while preserving which patient is in each group
        X_train, X_test, y_train,y_test,groups_train = Train_Test_Split(binned_data, debug=False,raw=False)

        if smote:
            #Equal *spaced* bins with SMOTE applied
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            # plt.hist(y_train_resampled, edgecolor='black', alpha=0.7)
            # Extend `groups_train` to match SMOTE's new sample count
            num_new_samples = len(X_train_resampled) - len(X_train)
            # Assign synthetic samples a placeholder group (-1)
            groups_resampled = np.concatenate([groups_train, np.full(num_new_samples, -1)])
            X_train, y_train, groups_train = X_train_resampled, y_train_resampled, groups_resampled

        if method == "ML":
            print(f"Method chosen is ML, raw = {raw}")
            #Run grid search on the split data **for multicategorical classifier**
            # grid_search = Optimize_Model(X_train,y_train,groups_train,debug=False,pkl_name=label)
            grid_search = Optimize_Model_Repeated_Iters(X_train,y_train,groups_train,debug=False,pkl_name=label)
            # with open('best_grid_search_10_bins_smote.pkl', 'rb') as f:
            #     grid_search = pickle.load(f)


            # #Sanity check our best model with a confusion matrix on X_test
            # y_pred = grid_search.best_estimator_.predict(X_test)
            # model_name = grid_search.best_estimator_.named_steps['classifier'].__class__.__name__
            # fig, ax = plt.subplots(figsize=(6, 5))
            # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            # ax.set_title(f"Best Model ({model_name}) on Test Data for {label}")
            # plt.savefig(f"cm-{label}")
            # plt.close('all')
            # plt.show()


            # run_visualizations(grid_search,X_train,y_train, X_test, y_test,bin_threshold=fifty_threshold,label=label)
        if method == "60/60":
            # Get predictions
            y_pred = sixty_sixty_predictions(X_test)
            y_true_binary = (y_test <= 7).astype(int)

            # Accuracy
            accuracy = accuracy_score(y_true_binary, y_pred)
            print(f"Accuracy: {accuracy:.2f}")

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_true_binary, y_pred)

            conf_matrix = ((conf_matrix/len(y_pred) ) * 100).astype(int)
            print("Confusion Matrix:")
            print(conf_matrix)
            labels = np.array([["{:.1f}%".format(val) for val in row] for row in conf_matrix])

            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=labels, fmt="", cmap="Blues", cbar=False,
                        xticklabels=["Predicted Non-Candidate", "Predicted Candidate"],
                        yticklabels=["Actual Non-Candidate", "Actual Candidate"])
            plt.xlabel("Prediction")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()
            y_true_binary = y_true_binary.values.tolist()
            print(y_true_binary)
            print(y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
            precision = precision_score(y_true_binary, y_pred, zero_division=0)
            recall = recall_score(y_true_binary, y_pred, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred, zero_division=0)
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # same as recall
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            print(f"Precision: {precision}, Recall: {recall}, F1: {recall}, Sensitivity:{sensitivity}, Specificity: {specificity}")




def Post_Iter_Processing():
    # Load the combined predictions


    folder = Path('bins-pkls')
    iter_info = []
    for pkl_file in folder.glob('*.pkl'):
        print(f"Loading: {pkl_file.name}")
        df = pd.read_pickle(f"bins-pkls/{pkl_file.name}")
        print(df['BestParams'])
        ModelName = df['ModelName'][0]
        if ModelName == 'XGClassifer':
            ModelName = 'XGBoostClassifier'

        # --- Aggregate across folds ---
        summary = df.groupby(['ModelName']).agg(
            f1_mean=('F1', 'mean'),
            f1_std=('F1', 'std'),
            auc_mean=('AUC', 'mean'),
            auc_std=('AUC', 'std'),
            n_folds=('AUC', 'count')
        ).reset_index()

        # --- Compute 95% confidence intervals ---
        summary['f1_95ci'] = t.ppf(0.975, summary['n_folds'] - 1) * summary['f1_std'] / np.sqrt(summary['n_folds'])
        summary['auc_95ci'] = t.ppf(0.975, summary['n_folds'] - 1) * summary['auc_std'] / np.sqrt(summary['n_folds'])

        # --- Sort by AUC or F1 as needed ---
        summary_sorted = summary.sort_values('auc_mean', ascending=False)

        # --- Display summary ---
        first_row = \
        summary_sorted[['ModelName', 'f1_mean', 'f1_std', 'f1_95ci', 'auc_mean', 'auc_std', 'auc_95ci']].iloc[:3]
        iter_info.append(summary_sorted.copy())
        # Print the full first row as a readable string
        print("\n Best Model Summary (Top Row):")
        print(first_row.to_string())


        # Sum all values
        totals = df[['tp', 'tn', 'fp', 'fn']].sum()
        tp, tn, fp, fn = totals['tp'], totals['tn'], totals['fp'], totals['fn']

        # Confusion matrix: actual rows, predicted columns
        conf_matrix = np.array([[tn, fp],
                                [fn, tp]])

        # Convert to percent
        total = conf_matrix.sum()
        percent_matrix = conf_matrix / total * 100
        percent_matrix = np.round(percent_matrix, 1)

        # Define annotations and custom colors
        labels = np.array([
            [f"{percent_matrix[0, 0]}%", f" {percent_matrix[0, 1]}%"],
            [f"{percent_matrix[1, 0]}%", f" {percent_matrix[1, 1]}%"]
        ])

        # Custom color matrix: light red for FP/FN, green for TP/TN
        colors = np.array([
            ['#a8e6a1', '#f4cccc'],  # TN, FP
            ['#f4cccc', '#a8e6a1']  # FN, TP
        ])

        # Plot with custom colored cells
        fig, ax = plt.subplots(figsize=(6, 4))

        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[i, j]))
                ax.text(j + 0.5, i + 0.5, labels[i, j],
                        ha='center', va='center', fontsize=12, fontweight='bold')

        # Formatting
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(['Predicted Non-Candidate', 'Predicted Candidate'])
        ax.set_yticklabels(['Actual Non-Candidate', 'Actual Candidate'])
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.invert_yaxis()
        ax.set_title(f"{ModelName}", fontsize=14)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'TRIO-figs/{ModelName}-cm.png')
        # plt.show()

    # summary_df = pd.concat(iter_info, ignore_index=True)
    # print(summary_df.head())
    # summary_df.to_excel("Sumamry_Iter.xlsx")




def Demographics_Table():
    df = pd.read_csv('candidacy_v3.csv')

    # Columns to summarize
    iqr_median = [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'HLdur_L', 'HLdur_R',
        'Hearing_Aid_Use_Time_L', 'Hearing_Aid_Use_Time_R',
        'CNC_L', 'CNC_R'
    ]

    category_percent = ['Gender', 'Race', 'Etiology_R', 'Etiology_L']

    # Ensure numeric columns are numeric
    df[iqr_median] = df[iqr_median].apply(pd.to_numeric, errors='coerce')

    # Compute median and IQR
    summary_stats = pd.DataFrame(columns=['Feature', 'Median', 'IQR', 'Missing'])

    # categorical_stats = pd.DataFrame(columns = ['Feature'])

    for col in iqr_median:
        median = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = str(q1) + '-' + str(q3)
        missing = df[col].isna().sum()
        summary_stats = summary_stats.append(
            {'Feature': col, 'Median': median, 'IQR': iqr, 'Missing': missing},
            ignore_index=True
        )
    # Show numeric summaries
    print("Median & IQR for numeric features:\n")
    print(summary_stats)

    # summary_stats.to_excel('summary-demographics.xlsx')

    # Compute category percentages
    print("\nCategory breakdowns:\n")
    for col in category_percent:
        print(f"--- {col} ---")
        counts = df[col].value_counts(dropna=False)
        percent = df[col].value_counts(normalize=True, dropna=False) * 100
        print(percent.round(2).astype(str))
        print(counts)






def nested_cross_optimize(n_iters=1,k_outer=10,k_inner=10,all_labels=[],raw_preds=False,smote=False):

    if smote:
        pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', preprocessing.StandardScaler()),  # Step 1: Scale features
        ('classifier', RandomForestClassifier(random_state=42))  # Step 2: Train classifier
        ])
    else:
       pipeline = Pipeline([
           ('scaler', preprocessing.StandardScaler()),  # Step 1: Scale features
           ('classifier', RandomForestClassifier(random_state=42))  # Step 2: Train classifier
       ])



    # Define parameter grids for different classifiers

    scoring = 'f1_micro'
    param_grid = {
        'RandomForestClassifier':[
        {  # Random Forest
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__max_depth': [10, 20],
            'classifier__min_samples_split': [2, 5]
        }],
        'XGClassifer':
        [{  # XGBoost
            'classifier': [XGBClassifier(eval_metric='logloss', random_state=42)],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1]
        }],
        'LogisticRegression':
        [{  # Vanilla Logistic Regression (no hyperparameter tuning)
            'classifier': [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)]
            # No hyperparameters specified
        }]
    }

    outer_cv = GroupKFold(n_splits=k_outer)
    inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=1)
    filtered_dataset = Clean_Data(debug=False, all_labels=all_labels)
    left_right_data = Create_Left_Right_Data(filtered_dataset, debug=False)
    if not raw_preds:
        binned_data, fifty_threshold = Add_Categorical_Bins(left_right_data, num_bins=10, debug=False)
        X = binned_data.drop(columns=['CNC_bin', 'CNC', 'patient_id'])
        y = binned_data['CNC_bin']
        groups = binned_data['patient_id'].values

    else:
        X = left_right_data.drop(columns=['CNC','patient_id'])
        y = left_right_data['CNC']
        y = (y < 50).astype(int)
        groups = left_right_data['patient_id'].values



    for model_name, param_subset in param_grid.items():
        print(f"Running procedure for {model_name}")


        iter_results = []
        for iter in tqdm(range(0,n_iters)):

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
                print(f"Iteration {iter}, fold {fold}")

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                groups_train = groups[train_idx]

                # Inner loop GridSearch
                clf = GridSearchCV(
                    pipeline,
                    param_grid=param_subset,
                    scoring=scoring,
                    cv=inner_cv,
                    n_jobs=-1,
                    verbose=3
                )
                clf.fit(X_train, y_train, groups=groups_train)

                #"Best model" is now the best estimator for the outer fold
                best_model = clf.best_estimator_
                y_pred = best_model.predict(X_test)
                y_pred_prob = best_model.predict_proba(X_test)

                #Make Binary
                if not raw_preds:
                    y_true_binary = (y_test <= 6).astype(int)
                    y_pred_binary = (y_pred <= 6).astype(int)

                    pred_prob_below_thresh = y_pred_prob[:, :7].sum(axis=1)
                    y_pred_prob_binary = (pred_prob_below_thresh >= 0.5).astype(
                        int)  # We classify as "1" if the summed probability is >=.5
                if raw_preds: #Already binarized
                    y_true_binary = y_test
                    y_pred_binary = y_pred
                    y_pred_prob_binary = y_pred_prob[:, 1]

                #Compute Metrics for the fold
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob_binary)
                roc_auc = auc(fpr, tpr)

                # Metrics: Binarize
                f1 = f1_score(y_true_binary, y_pred_binary, average='macro')


                tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # same as recall
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

                #Append to the larger list, which is faster than appending dfs
                new_row = {
                    'ModelName': model_name,
                    'BestParams': clf.best_params_,  # dict of best hyperparameters found
                    'Iter': iter,
                    'Fold': fold + 1,
                    'AUC': roc_auc,
                    'F1': f1,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'Precision': precision,
                    'Recall': recall,
                    'tp':tp,
                    'tn':tn,
                    'fp':fp,
                    'fn':fn,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_prob

                }
                print(new_row)
                iter_results.append(new_row)



        all_iter_results = pd.DataFrame(iter_results)

        all_iter_results.to_pickle(f'{model_name}_nested_raw2bi.pkl')

        print(all_iter_results.head())

        print(f"Shape of final dataframe: {all_iter_results.shape}")










if __name__ == '__main__':
    # different_variables_run()
    # sixty_sixty_run()
    # PyCaret_Run_Binary()
    # different_variables_run(full=True)
    # Post_Iter_Processing()
    # Demographics_Table()
    all_labels =  [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
    ]
    nested_cross_optimize(n_iters=1, k_outer=2, k_inner=2, all_labels=all_labels,raw_preds=True)







