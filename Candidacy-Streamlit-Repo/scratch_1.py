import pandas as pd
from sklearn.model_selection import GroupKFold,GridSearchCV,GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE

def Clean_Data(debug=False): #This function extracts the columns of interest, removes NA, and adds patient ID
    full_dataset = pd.read_csv("C:/Users/jawad/Downloads/Streamlit_ML_Candidacy/Candidacy-Streamlit-Repo/candidacy_v3.csv")

    all_labels = [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R','hz6000_L','hz8000_R','hz8000_L',
        'WRS_L','WRS_R','Age','CNC_L', 'CNC_R'
    ]

    #Need to keep everything together to prevent loss of data
    filtered_dataset = full_dataset[all_labels].dropna()
    filtered_dataset['patient_id'] = range(1, len(filtered_dataset) + 1)
    print(filtered_dataset.shape)

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
    ages = df['Age'].copy()

# Rename columns by removing the side-specific suffix
    left_df.columns = left_df.columns.str.replace('_L$', '', regex=True)
    right_df.columns = right_df.columns.str.replace('_R$', '', regex=True)

    # Concatenate the dataframes
    left_df['patient_id'] = patient_ids
    right_df['patient_id'] = patient_ids

    #Add in age as well
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


def Train_Test_Split(binned_data,debug=False):
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
def Optimize_Model(X_train,y_train,groups_train,debug=False):
    '''This section will take the selected model in 'params' and train the model'''

    kf = GroupKFold(n_splits=5)
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
            'classifier': [XGBClassifier(eval_metric='logloss', random_state=42)],
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
def run_visualizations(grid_search,X_test,y_test,bin_threshold):
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
        y_pred_prob_binary = (pred_prob_below_thresh >= 0.5).astype(int)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob_binary)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC AUC Curve (Threshold = {bin_threshold})")
        plt.legend()
        plt.grid()

        # Bootstrap ROC and get confidence bounds
        fpr,tpr, tpr_lower, tpr_upper = bootstrap_roc_auc(
            y_true_binary, y_pred_prob_binary
        )
        # Compute AUC for the lower bound of the TPR
        auc_lower_bound = auc(fpr, tpr_lower)

        # Compute AUC for the upper bound of the TPR
        auc_upper_bound = auc(fpr, tpr_upper)
        # roc_auc = auc(fpr, mean_tpr)
        # plt.plot(fpr, mean_tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        plt.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.2)
        model_performance.append({
            'Model': model_name,
            'AUC': roc_auc,
            'CI Lower Bound': auc_lower_bound,  # At FPR = 0
            'CI Upper Bound': auc_upper_bound # At FPR = 0
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
    performance_df.to_excel("performance.xlsx")

    ##Calibration curve
    plt.figure(figsize=(8, 6))

    for model_name, model_info in best_models.items():
        best_model = model_info['model']
        print(f"Running model: {model_name}")
        y_true_binary = (y_test <= bin_threshold).astype(int)
        y_pred_prob = best_model.predict_proba(X_test)  #Outputs an array of probabilities for each test data corresponding to the ten categories
        # Classes to sum for binary prob
        pred_prob_below_thresh = y_pred_prob[:, :bin_threshold].sum(axis=1)
        y_pred_prob_binary = (pred_prob_below_thresh >= 0.5).astype(int)

        # Get the calibration curve (fraction of positives vs. mean predicted probability)
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_binary, pred_prob_below_thresh, n_bins=10)

        # Plot the calibration curve
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name} Calibration Curve')
        if "Random" in model_name:
            counts, bins = np.histogram(pred_prob_below_thresh, bins=10, range=(0, 1))
            proportions = counts / counts.sum()
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_width = bins[1] - bins[0]
            plt.bar(bin_centers, proportions, width=bin_width, alpha=0.5, label='RandomForest Histogram')
            plt.ylim(0, 1)  # Optional: keep Y-axis from 0 to 1
            # plt.hist(pred_prob_below_thresh/pred_prob_below_thresh.sum(),bins=10, alpha=0.3, label=f'{model_name} Histogram')

    # Plot the ideal calibration line (diagonal line where predicted probability matches actual frequency)
    plt.plot([0, 1], [0, 1], color='black', linewidth=2, label='Perfectly calibrated')

    # Final plot settings
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves for Different Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # num_bins = 10
    # smote = False
    #
    # #############################
    # #Extract columns of interest, remove NA, and add patient ID and CNC bin
    # filtered_dataset = Clean_Data(debug=False)
    # left_right_data = Create_Left_Right_Data(filtered_dataset,debug=False)
    # binned_data,fifty_threshold = Add_Categorical_Bins(left_right_data,num_bins=10,debug=True)
    # print(binned_data.head)
    # # Split into train/test while preserving which patient is in each group
    # X_train, X_test, y_train,y_test,groups_train = Train_Test_Split(binned_data, debug=True)
    #
    # if smote:
    #     #Equal *spaced* bins with SMOTE applied
    #     smote = SMOTE(sampling_strategy="auto", random_state=42)
    #     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    #     plt.hist(y_train_resampled, edgecolor='black', alpha=0.7)
    #     # Extend `groups_train` to match SMOTE's new sample count
    #     num_new_samples = len(X_train_resampled) - len(X_train)
    #     # Assign synthetic samples a placeholder group (-1)
    #     groups_resampled = np.concatenate([groups_train, np.full(num_new_samples, -1)])
    #     X_train, y_train, groups_train = X_train_resampled, y_train_resampled, groups_resampled
    # #Run grid search on the split data **for multicategorical classifier**
    # grid_search = Optimize_Model(X_train,y_train,groups_train,debug=True) #saves to best_grid_search_4_9.pkl
    # # with open('best_grid_search_10_bins_smote.pkl', 'rb') as f:
    # #     grid_search = pickle.load(f)
    #
    # plt.close('all')
    # #Sanity check our best model with a confusion matrix on X_test
    # y_pred = grid_search.best_estimator_.predict(X_test)
    # print("XTest",X_test)
    # model_name = grid_search.best_estimator_.named_steps['classifier'].__class__.__name__
    # fig, ax = plt.subplots(figsize=(6, 5))
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    # ax.set_title(f"Best Model ({model_name}) on Test Data")
    # plt.show()
    #
    #
    # run_visualizations(grid_search, X_test, y_test,bin_threshold=fifty_threshold)

    #### Comparing weird model differences
    import joblib

    grid1 = joblib.load('C:/Users/jawad/Downloads/Streamlit_ML_Candidacy/Candidacy-Streamlit-Repo/grid_search.pkl')
    grid2 = joblib.load('C:/Users/jawad/Downloads/Streamlit_ML_Candidacy/Candidacy-Streamlit-Repo/best_grid_search_10_bins_percentiles.pkl')

    print("Best Estimator - Grid 1:\n", grid1.best_estimator_)
    print("\nBest Estimator - Grid 2:\n", grid2.best_estimator_)

    print("\nBest Parameters - Grid 1:\n", grid1.best_params_)
    print("\nBest Parameters - Grid 2:\n", grid2.best_params_)

    print("\nBest Score - Grid 1:", grid1.best_score_)
    print("Best Score - Grid 2:", grid2.best_score_)
    cv_results_1 = pd.DataFrame(grid1.cv_results_)
    cv_results_2 = pd.DataFrame(grid2.cv_results_)

    # View top-ranked configurations
    print("\nTop 5 Grid 1 Results:")
    print(cv_results_1.sort_values(by="mean_test_score", ascending=False).head())

    print("\nTop 5 Grid 2 Results:")
    print(cv_results_2.sort_values(by="mean_test_score", ascending=False).head())
    param_keys_1 = set(cv_results_1.columns)
    param_keys_2 = set(cv_results_2.columns)

    print("\nParameter keys unique to Grid 1:", param_keys_1 - param_keys_2)
    print("Parameter keys unique to Grid 2:", param_keys_2 - param_keys_1)

    from sklearn.metrics import classification_report

    all_labels = [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
    ]
    filtered_dataset = Clean_Data(debug=False)
    left_right_data = Create_Left_Right_Data(filtered_dataset, debug=False)
    binned_data, fifty_threshold = Add_Categorical_Bins(left_right_data, num_bins=10, debug=True)
    # Split into train/test while preserving which patient is in each group
    X_train, X_test, y_train, y_test, groups_train = Train_Test_Split(binned_data, debug=True)

    y_pred1 = grid1.predict(X_test)
    y_pred2 = grid2.predict(X_test)
    print("\nPerformance on X_test - Grid 1:")
    print(classification_report(y_test, y_pred1))

    print("\nPerformance on X_test - Grid 2:")
    print(classification_report(y_test, y_pred2))

    run_visualizations(grid1, X_test, y_test, 7)

    run_visualizations(grid2, X_test, y_test, 7)











#Train the model on a series of different types including RandomForest, XGBoost, and Logistic Regression
#These should use k-fold as the method of holdout
#Moreover, we want all of these models to be ORDINAL only.