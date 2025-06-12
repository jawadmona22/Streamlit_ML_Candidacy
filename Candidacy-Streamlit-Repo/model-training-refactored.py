import pandas as pd
from sklearn.model_selection import GroupKFold,GridSearchCV,GroupShuffleSplit
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

def Clean_Data(debug=False,all_labels=[]): #This function extracts the columns of interest, removes NA, and adds patient ID
    full_dataset = pd.read_csv("candidacy_v3.csv")

    if 'Age' in full_dataset.columns:
        # Calculate median age (ignores NaNs by default)
        median_age = full_dataset["Age"].median()
        print("Missing before:", full_dataset["Age"].isna().sum())
        full_dataset["Age"].fillna(median_age, inplace=True)
        print("Missing after:", full_dataset["Age"].isna().sum())

    #Need to keep everything together to prevent loss of data
    filtered_dataset = full_dataset[all_labels].dropna()
    filtered_dataset['patient_id'] = range(1, len(filtered_dataset) + 1)

    if debug:
        print(f"Filtered Dataset: {filtered_dataset.head}")
        print(f"Filtered Columns: {filtered_dataset.columns}")


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

def Add_Categorical_Bins(left_right_data,num_bins,debug):
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



def Optimize_Model(X_train,y_train,groups_train,debug=False,raw=False,soft_label=False,pkl_name = 'grid_search'):
    '''This section will take the selected model in 'params' and train the model'''

    kf = GroupKFold(n_splits=5)
    if soft_label == True and raw == True:
        # Define parameter grids
        pipeline = Pipeline([
            ('regressor', RandomForestRegressor())  # Placeholder
        ])
        param_grid = [
            {  # Random Forest Regressor
                'regressor': [RandomForestRegressor(random_state=42)],
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [10, 20, None],
                'regressor__min_samples_split': [2, 5, 10]
            },
            {  # XGBoost Regressor
                'regressor': [XGBRegressor(objective='reg:squarederror', random_state=42)],
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [3, 6, 10],
                'regressor__learning_rate': [0.01, 0.1, 0.2]
            },
            {  # Vanilla Linear Regression (no tuning)
                'regressor': [LinearRegression()]
            }
        ]

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # works for both regressors and log-loss classifiers
            cv=kf,
            n_jobs=-1,
            verbose=3
        )

        grid_search.fit(X_train, y_train, groups=groups_train)

        with open(f'{pkl_name}.pkl', 'wb') as f:
            pickle.dump(grid_search, f)

        if debug:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            best_model = grid_search.best_estimator_  # or gs_xgb.best_estimator_
            y_pred = best_model.predict(X_train)
            plt.figure(figsize=(6, 6))
            plt.scatter(y_train, y_pred, alpha=0.4, s=10)
            plt.plot([0, 1], [0, 1], 'r--', label="Perfect Prediction")
            plt.xlabel("True Soft Label")
            plt.ylabel("Predicted Soft Label")
            plt.title(f"Model – Soft Label Prediction Accuracy (Training)")
            plt.legend()
            plt.grid(True)
            plt.show()
        return grid_search

    if raw == True:
        # Define parameter grids
        pipeline = Pipeline([
            ('regressor', RandomForestRegressor())  # Placeholder
        ])
        param_grid = [
            {  # Random Forest Regressor
                'regressor': [RandomForestRegressor(random_state=42)],
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [10, 20, None],
                'regressor__min_samples_split': [2, 5, 10]
            },
            {  # XGBoost Regressor
                'regressor': [XGBRegressor(objective='reg:squarederror', random_state=42)],
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [3, 6, 10],
                'regressor__learning_rate': [0.01, 0.1, 0.2]
            },
            {  # Vanilla Linear Regression (no tuning)
                'regressor': [LinearRegression()]
            }
        ]

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # works for both regressors and log-loss classifiers
            cv=kf,
            n_jobs=-1,
            verbose=3
        )

        grid_search.fit(X_train, y_train, groups=groups_train)

        with open('best_grid_search_raw_mixed.pkl', 'wb') as f:
            pickle.dump(grid_search, f)

        if debug:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # Predict on training data
            y_train_pred = grid_search.best_estimator_.predict(X_train)

            # Calculate metrics
            mse = mean_squared_error(y_train, y_train_pred)
            mae = mean_absolute_error(y_train, y_train_pred)
            r2 = r2_score(y_train, y_train_pred)

            print(f"Training MSE: {mse:.4f}")
            print(f"Training MAE: {mae:.4f}")
            print(f"Training R^2: {r2:.4f}")

            # Scatter plot: Predicted vs Actual
            plt.figure(figsize=(8, 6))
            plt.scatter(y_train, y_train_pred, alpha=0.6)
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predicted vs Actual (Training Data)")
            plt.grid(True)
            plt.show()

            # Residual plot: Residuals vs Predicted
            residuals = y_train - y_train_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_train_pred, residuals, alpha=0.6)
            plt.hlines(0, y_train_pred.min(), y_train_pred.max(), colors='r', linestyles='dashed')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals (Actual - Predicted)")
            plt.title("Residual Plot (Training Data)")
            plt.grid(True)
            plt.show()
        return grid_search

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
                'classifier': [XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 6, 10],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            },
            # {  # Logistic Regression
            #     'classifier': [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)],
            #     'classifier__C': [0.01, 0.1, 1, 10],
            #     'classifier__penalty': ['l2']
            # },
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
            cv=kf,
            n_jobs=-1,
            verbose=3
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        with open('best_grid_search_10_bins_percentiles.pkl', 'wb') as f:
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
def run_visualizations(grid_search,X_test,y_test,bin_threshold,raw=False,label='Default'):
    if raw == False:
        best_models = {}
        model_performance = []



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
            model_pipeline.fit(X_test, y_test)
            classifier = model_pipeline.named_steps['classifier']
            best_models[model_name]['model'] = classifier  # Store fitted classifier

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
            plt.close('all')
            plt.figure()
            # Plot the ROC curve
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC AUC Curve (Threshold = {bin_threshold})")
            plt.legend()
            plt.savefig(f"{label}-AUC.png")

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
        plt.show()
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
            ax.bar(bin_centers, proportions, width=bin_width, alpha=0.5, label='Pred Prob Histogram')
            ax.set_ylim(0, 1)

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
        plt.show()

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


def different_variables_run():
    ##Run 1

    set_dict = {
        "ag-only":
        #Audiogram Only
        [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L', 'CNC_L', 'CNC_R'
        ],
        "wrs":
        #audiogram + WRS
        [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'CNC_L', 'CNC_R'
        ],
        "wrs-a":

        #Audiogram + WRS + Age
        [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age','CNC_L', 'CNC_R'
        ],
        # #TODO: Audiogram + WRS + Age + Duration HL
        # "wrs-a-dhl":
        #
        #     [
        # 'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        # 'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        # 'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        # 'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        # 'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        # 'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
        # ],
        # "wrs-a-dhl-dha":
        # #TODO:Audiogram + WRS + Age + Duration HL + Duration HA Usage
        # [
        #     'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        #     'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        #     'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        #     'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        #     'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        #     'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
        # ]

    }

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
                grid_search = Optimize_Model(X_train, y_train_soft, groups_train, debug=True, raw=True,soft_label=is_soft_label,pkl_name=label)

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

    # Split into train/test while preserving which patient is in each group
        X_train, X_test, y_train,y_test,groups_train = Train_Test_Split(binned_data, debug=False,raw=False)

        if smote:
            #Equal *spaced* bins with SMOTE applied
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            plt.hist(y_train_resampled, edgecolor='black', alpha=0.7)
            # Extend `groups_train` to match SMOTE's new sample count
            num_new_samples = len(X_train_resampled) - len(X_train)
            # Assign synthetic samples a placeholder group (-1)
            groups_resampled = np.concatenate([groups_train, np.full(num_new_samples, -1)])
            X_train, y_train, groups_train = X_train_resampled, y_train_resampled, groups_resampled

        if method == "ML":
            #Run grid search on the split data **for multicategorical classifier**
            grid_search = Optimize_Model(X_train,y_train,groups_train,debug=True)
            # with open('best_grid_search_10_bins_smote.pkl', 'rb') as f:
            #     grid_search = pickle.load(f)


            #Sanity check our best model with a confusion matrix on X_test
            y_pred = grid_search.best_estimator_.predict(X_test)
            model_name = grid_search.best_estimator_.named_steps['classifier'].__class__.__name__
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            ax.set_title(f"Best Model ({model_name}) on Test Data for {label}")
            plt.savefig(f"cm-{label}")
            # plt.show()


            run_visualizations(grid_search, X_test, y_test,bin_threshold=fifty_threshold,label=label)
        if method == "60/60":
            # Get predictions
            y_pred = sixty_sixty_predictions(X_test)
            y_true_binary = (y_test <= 5).astype(int)

            # Accuracy
            accuracy = accuracy_score(y_true_binary, y_pred)
            print(f"Accuracy: {accuracy:.2f}")

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_true_binary, y_pred)
            print("Confusion Matrix:")
            print(conf_matrix)

            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Predicted Non-Candidate", "Predicted Candidate"],
                        yticklabels=["Actual Non-Candidate", "Actual Candidate"])
            plt.xlabel("Prediction")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()






if __name__ == '__main__':
    different_variables_run()



#Train the model on a series of different types including RandomForest, XGBoost, and Logistic Regression
#These should use k-fold as the method of holdout
#Moreover, we want all of these models to be ORDINAL only.