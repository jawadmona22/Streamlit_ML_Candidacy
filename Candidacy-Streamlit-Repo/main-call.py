import pandas as pd
from sklearn.calibration import calibration_curve
import math
import optimize_model
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, confusion_matrix
import classifier_pipeline as cp
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE



if not hasattr(math, 'prod'):
    math.prod = np.prod
# def plot_calibration_curve(model_name, y_true, y_probs, ax):
#     # Compute calibration curve
#     fraction_of_positives, mean_predicted_value = calibration_curve(
#         y_true, y_probs, n_bins=10, strategy='uniform'
#     )
#     ax.plot(mean_predicted_value, fraction_of_positives, marker='o', label=model_name)
#     ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
#     ax.set_xlabel('Mean Predicted Probability')
#     ax.set_ylabel('Fraction of Positives')
#     ax.set_title(f'Calibration Curve for {model_name}')
#     ax.legend()

def bootstrap_predict_with_cv_validation(X, y, n_splits=5, n_bootstrap=1, threshold=40):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_probabilities = []
    true_labels = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        fold_probabilities = []

        # Progress bar for bootstrapping predictions
        for i, x_test_instance in tqdm(enumerate(X_test.values), total=X_test.shape[0], desc="Bootstrapping"):
            # Collect bootstrapped predictions for this test instance
            test_predictions = []
            for j in range(n_bootstrap):
                X_resampled, y_resampled = resample(X_train, y_train, random_state=j)
                model = RandomForestRegressor()
                model.fit(X_resampled, y_resampled)
                y_pred = model.predict(X_test)
                test_predictions.append(y_pred[i])

            # Probability of being below threshold for this instance
            prob_below_threshold = (np.array(test_predictions) < threshold).mean()
            fold_probabilities.append(prob_below_threshold)

        all_probabilities.extend(fold_probabilities)
        true_labels.extend((y_test < threshold).astype(int).values)  # Binary labels

    # Calculate Brier Score for validation
    brier_score = brier_score_loss(true_labels, all_probabilities)
    print(f"\nBrier Score for probability predictions: {brier_score:.4f}")

    # Visualization of predicted probabilities vs true labels
    plt.figure(figsize=(10, 6))
    plt.hist([p for i, p in enumerate(all_probabilities) if true_labels[i] == 1], bins=10, alpha=0.5, label="True < 40")
    plt.hist([p for i, p in enumerate(all_probabilities) if true_labels[i] == 0], bins=10, alpha=0.5,
             label="True >= 40")
    plt.xlabel("Predicted Probability of CNC < 40")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Predicted Probabilities of CNC < 40 by True Label")
    plt.show()

    return brier_score





def SetFeaturesAndTarget(full_dataset, columns_of_interest):
    print("Setting Features, Target")
    # Filter the dataset to only include the columns of interest
    filtered_data = full_dataset[columns_of_interest]
    filtered_data = filtered_data.dropna()
    # Separate features and target
    features = filtered_data.iloc[:, :-2]
    target = filtered_data.iloc[:, -2:]

    return features, target

def CreateCombinedDataset(data,num_bins,plot=False):
    # Original DataFrame
    df = pd.DataFrame(data)

    # Separate L and R columns
    left_df = df.filter(regex='_L$').copy()
    right_df = df.filter(regex='_R$').copy()

    # Extract the patient_id column
    patient_ids = df['patient_id'].copy()

    # Rename columns by removing the side-specific suffix
    left_df.columns = left_df.columns.str.replace('_L$', '', regex=True)
    right_df.columns = right_df.columns.str.replace('_R$', '', regex=True)

    # Concatenate the dataframes
    left_df['patient_id'] = patient_ids
    right_df['patient_id'] = patient_ids

    new_df = pd.concat([left_df, right_df], ignore_index=True)


    # Apply ordinal binning (if needed)
    binned_df, bin_threshold = CreateOrdinal(new_df,num_bins=num_bins,plot=plot)

    return binned_df, bin_threshold



def train_and_save_model(features, target,X_train_scaled, X_test_scaled, y_train, y_test):
    #begin model optimization
    study = optimize_model.optimize_model(features, target)
    best_model = optimize_model.RebuildBestModel(study)
    best_model.fit(X_train_scaled, y_train)
    print(type(best_model))

    # Evaluate the model on the test data
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE of the best model: {mse}")
    joblib.dump(best_model, "best_model_100epochs-percentile.pkl")


def CreateOrdinal(data, num_bins=15, plot=True,type='Equal Space'):
    """
    Bins numeric data into percentile-based bins with numeric labels for the data
    and range-based labels for visualization.

    Parameters:
        data (DataFrame): The input DataFrame containing the column 'CNC' to bin.
        num_bins (int): Number of bins to divide the data into.
        plot (bool): If True, plot the histogram of binned values with range labels.

    Returns:
        DataFrame: The input DataFrame with an added 'CNC_bin' column.
        int: Bin threshold for bins below 40.
    """
    # Calculate percentile thresholds
    percentiles = np.linspace(0,100,10+1)
    thresholds = np.percentile(data['CNC'],percentiles)
    data['CNC_bin'] = np.digitize(data['CNC'], thresholds, right=False)
    print(np.sort(data['CNC']))
    # Identify bin threshold below 50
    bin_threshold = max(i for i, upper in enumerate(thresholds[1:]) if upper < 50)
    # Plot if needed
    if plot:
            plt.figure(figsize=(10, 6))
            plt.hist(data['CNC'], bins=thresholds, edgecolor='black', alpha=0.7)
            plt.title('Distribution of CNC Bins (Percentile-based)')
            plt.xlabel('CNC Score')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    return data, bin_threshold


def plot_calibration_curve(y_true, y_pred_prob, num_bins):
    """
    Plots the calibration curve for a given set of true labels and predicted probabilities.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy='uniform')

    plt.plot(prob_pred, prob_true, marker='o', label=f'Bins={num_bins}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title(f'Calibration Curve (Bins={num_bins})')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.grid(alpha=0.6)
    plt.legend()

def evaluate_bins_with_auc(num_bins, full_dataset, grid_search,smote=False):
        combined_sides_data, bin_threshold = CreateCombinedDataset(full_dataset, num_bins=num_bins, plot=False)
        X = combined_sides_data.drop(columns=['CNC_bin', 'CNC', 'patient_id'])
        y = combined_sides_data['CNC_bin']
        groups = combined_sides_data['patient_id'].values

        # Train-test split
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print("Class distribution before SMOTE:", np.bincount(y_train))
        groups_train = groups[train_idx]  # Extract corresponding train groups


        # Apply SMOTE
        if smote:
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Extend `groups_train` to match SMOTE's new sample count
            num_new_samples = len(X_train_resampled) - len(X_train)
            groups_resampled = np.concatenate([groups_train, np.full(num_new_samples, -1)])
            # Assign synthetic samples a placeholder group (-1)

            X_train, y_train, groups_train = X_train_resampled, y_train_resampled, groups_resampled

            print("Class distribution after SMOTE:", np.bincount(y_train))

        # Train and evaluate
        grid_search.fit(X_train, y_train, groups=groups_train)
        best_model = grid_search.best_estimator_ #best model
        if num_bins == 11:
            joblib.dump(best_model, 'best_model_percentiles_11bins-smote.pkl')

        if num_bins == 10:
            joblib.dump(best_model, 'best_model_percentiles_10bins-smote.pkl')

        #Binary classification for ROC AUC
        y_true_binary = (y_test <= bin_threshold).astype(int)
        positive_classes = np.where(np.unique(y_test) <= bin_threshold)[0] #Classes under the 40% threshold
        y_pred_prob = best_model.predict_proba(X_test)[:,positive_classes].sum(axis=1) #Sums probabilities for
        fpr,tpr,_ = roc_curve(y_true_binary,y_pred_prob)
        roc_auc = auc(fpr,tpr) #This is the raw score

        return grid_search.best_score_, roc_auc, fpr,tpr,y_true_binary,y_pred_prob


def binary_search_optimal_bins(full_dataset, grid_search, low=5, high=25, tolerance=1): #Returns information on the bin with the lowest MAE and best ROC AUC
        best_bins = None
        best_score = -float('inf')
        roc_auc_values = []
        roc_data_plotting = []
        calibration_data = []  # Store calibration data for plotting

        mae_results = {}
        while high - low > tolerance:
            mid = (low + high) // 2
            mid_score,mid_roc_auc,mid_fpr,mid_tpr,y_true_binary_mid,y_pred_prob_mid = evaluate_bins_with_auc(mid, full_dataset, grid_search,smote=True)
            mid_plus_score,mid_plus_roc_auc,mid_plus_fpr,mid_plus_tpr,y_true_binary_midplus,y_pred_prob_midplus = evaluate_bins_with_auc(mid + 1, full_dataset, grid_search,smote=True)
            if not any(item[0] == mid for item in roc_data_plotting):
                #Track ROC and MAE data
                roc_data_plotting.append((mid,mid_fpr,mid_tpr,mid_roc_auc,y_true_binary_mid,y_pred_prob_mid))
                roc_auc_values.append((mid, mid_roc_auc))
                mae_results[mid] = mid_score

            if not any(item[0] == mid+1 for item in roc_data_plotting):
                #Track ROC and MAE data
                roc_data_plotting.append((mid+1,mid_plus_fpr,mid_plus_tpr,mid_plus_roc_auc,y_true_binary_midplus,y_pred_prob_midplus))
                roc_auc_values.append((mid + 1, mid_plus_roc_auc))
                mae_results[mid +1] = mid_plus_score



            # Store calibration data
            if not any(item[0] == mid for item in calibration_data):
                calibration_data.append((mid, y_true_binary_mid, y_pred_prob_mid))
                print(f"Y Pred Prob: {y_pred_prob_mid}")

            if not any(item[0] == mid + 1 for item in calibration_data):
                calibration_data.append((mid + 1, y_true_binary_midplus, y_pred_prob_midplus))

            # Compare scores and decide the direction
            if mid_score > mid_plus_score:
                high = mid  # Focus on the lower range
                if mid_score > best_score:
                    best_score = mid_score
                    best_bins = mid
            else:
                low = mid + 1  # Focus on the upper range
                if mid_plus_score > best_score:
                    best_score = mid_plus_score
                    best_bins = mid + 1

        return best_bins, best_score, roc_auc_values,roc_data_plotting, mae_results,calibration_data


# if __name__ == '__main__':

# #####Pre-Processing Data Setup#####
#
#     full_dataset = pd.read_csv("candidacy_v3.csv")
#     columns_of_interest = [  #Final string specifies target, all others are feature columns
#         'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
#         'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
#         'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
#         'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
#         'hz4000_R', 'hz4000_L', 'hz6000_R','hz6000_L','hz8000_R','hz8000_L',
#         'WRS_L','WRS_R','Age', 'CNC_L', 'CNC_R'
#     ]
#
#     #Split into features and target, but does not separate by ear to preserve patient ID
#     features, target = SetFeaturesAndTarget(full_dataset, columns_of_interest)   #target includes CNC_L and CNC_R, features include both
#     full_dataset = pd.concat([features, target], axis=1)
#     ##Add patient ID
#     full_dataset['patient_id'] = range(1, len(full_dataset) + 1)
#
#     kf = GroupKFold(n_splits=5)
#
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),  # Step 1: Standardize features
#         ('classifier', RandomForestClassifier(random_state=42))  # Step 2: Random Forest Classifier
#     ])
#     # Define parameter grid for RandomForest
#     param_grid = {
#         'classifier__n_estimators': [50, 100, 200],
#         'classifier__max_depth': [10, 20, None],
#         'classifier__min_samples_split': [2, 5, 10]
#     }
#
#     # Set up GridSearchCV with GroupKFold
#     grid_search = GridSearchCV(
#         pipeline,
#         param_grid=param_grid,
#         scoring='neg_mean_absolute_error',
#         cv=kf,
#         n_jobs=-1,
#         verbose=2
#     )
#
####Bin Optimization####
    optimal_bins, best_score,roc_auc_values,roc_data_plotting, mae_results,calibration_data = binary_search_optimal_bins(full_dataset, grid_search)
    print(f"Optimal number of bins: {optimal_bins}, with score: {best_score}")
    num_bins = "5-to-25"
    #Plot scores versus bins
    bins,aucs = zip(*roc_auc_values)
    plt.figure(figsize=(10,6))
    plt.scatter(bins, aucs)
    plt.xlabel('Number of Bins')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs. Number of Bins')
    plt.grid(alpha=0.7)
    plt.legend()
    plt.savefig(f'ROC-Scores-{num_bins}-bins.png')
    plt.show()
#
#     plt.figure(figsize=(10, 8))
#
#     mean_fpr = np.linspace(0,1,100) #Fixed FPR values, for interpolation
#
#     for num_bins, fpr, tpr, roc_auc,y_test,y_probs in roc_data_plotting:
#         n_bootstraps = 1000
#         rng = np.random.RandomState(42)
#
#         boot_tprs = []
#         for _ in range(n_bootstraps):
#             indices = rng.choice(len(y_test),len(y_test),replace=True)
#
#             # Ensure indices are within range and select elements one by one
#             y_test_boot = np.array([y_test[i] for i in indices if i < len(y_test)])
#             y_probs_boot = np.array([y_probs[i] for i in indices if i < len(y_probs)])
#
#             fpr_boot, tpr_boot, _ = roc_curve(y_test_boot,y_probs_boot)
#             tpr_interp = np.interp(mean_fpr, fpr_boot,tpr_boot)
#             boot_tprs.append(tpr_interp)
#         boot_tprs = np.array(boot_tprs)
#         tpr_lower = np.percentile(boot_tprs,2.5,axis = 0)
#         tpr_upper = np.percentile(boot_tprs,97.5,axis = 0)
#
#
#         plt.plot(fpr, tpr, label=f'Bins={num_bins} (AUC={roc_auc:.2f})')
#
#         plt.fill_between(mean_fpr,tpr_lower,tpr_upper,alpha = .2, label = f'Bins = {num_bins} 95% CI')
#     # Plot diagonal line (random classifier)
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
#
#     # Customize plot
#     plt.title('ROC AUC Curves for Different Bin Sizes')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc='lower right')
#     plt.grid(alpha=0.6)
#     plt.savefig(f'ROC-Curves-{num_bins}-bins.png')
#
#     plt.show()
#
#
#     plt.figure()
#     # Visualize results
#     plt.scatter(mae_results.keys(), mae_results.values(), marker='o')
#     plt.title('Performance vs. Number of Percentile Bins')
#     plt.xlabel('Number of Bins')
#     plt.ylabel('Neg Mean Absolute Error')
#     plt.grid()
#     plt.savefig(f'Mae-Scores.png')
#     plt.show()
#
#     # Final dataset with optimal bins
#     combined_sides_data, bin_threshold = CreateCombinedDataset(full_dataset, num_bins=10, plot=False)
#     print(f"Final dataset shape: {combined_sides_data.shape}")
#
#     #Calibration curves
#
#     # Plot calibration curves for all bins
#     plt.figure(figsize=(10, 8))
#
#     for bin_size, y_true_binary, y_pred_prob in calibration_data:
#         if np.any((y_pred_prob > 1)) or np.any((y_pred_prob < 0)):
#             print("BAD Y_PRED DETECTED")
#         else:
#             prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_prob, n_bins=10, strategy='uniform')
#             plt.plot(prob_pred, prob_true, marker='o', label=f'Bins={bin_size}')
#
#     # Perfect calibration line
#     plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
#
#     # Customize plot
#     plt.title('Calibration Curves for Different Bin Sizes')
#     plt.xlabel('Mean Predicted Probability')
#     plt.ylabel('Fraction of Positives')
#     plt.legend(loc='best')
#     plt.grid(alpha=0.6)
#     plt.savefig("Calibration-Plt.png")
#     plt.show()

###############################








    # # Train model using GridSearchCV
    # grid_search.fit(X_train, y_train,groups=groups[train_idx])
    #
    # # Get the best model from GridSearchCV
    # best_model = grid_search.best_estimator_
    # joblib.dump(best_model, 'best_model_percentiles.pkl')
    # # best_model = joblib.load('best_model_dec.pkl')
    #
    # # Evaluate the best model on the test set
    #
    # y_pred = best_model.predict(X_test)
    #
    # ##Visualizations
    # # bin_threshold = 8
    # y_true_binary = (y_test <= bin_threshold).astype(int)
    # y_pred_binary = (y_pred <= bin_threshold).astype(int)
    # cm = confusion_matrix(y_true_binary, y_pred_binary)
    # # Displaying the confusion matrix using a heatmap
    # # plt.figure(figsize=(8, 6))  # Optional: Size of the plot
    # cm = confusion_matrix(y_true_binary, y_pred_binary)
    #
    # # Plot the confusion matrix using seaborn heatmap
    # plt.figure(figsize=(8, 6))  # Optional: Size of the plot
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CNC > 40', 'CNC <=40'],
    #             yticklabels=['CNC > 40', 'CNC <=40'])
    # plt.title('Confusion Matrix: CNC40')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()
    #
    #
    # ##ROC-AUC curve
    # y_true_binary = (y_test <= bin_threshold).astype(int)
    #
    # # Get predicted probabilities for the positive class (bin 8 or lower)
    # y_pred_prob = best_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    #
    # # Compute ROC curve
    # fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_prob)
    # roc_auc = auc(fpr, tpr)
    #
    # # Plot ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # plt.hist(y_pred_prob, bins=20, range=(0, 1), histtype='step', lw=2, label='Predicted Probabilities')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Predicted Probabilities')
    # plt.legend()
    # plt.show()
    #
    #
    #

  #######


    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#     # best_model = joblib.load('best_model_100epochs.pkl')
#     best_model = XGBRegressor(objective='reg:squarederror', random_state=42)  # For regression task
#
#     #Kfold begins#################
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     mse_scores = []
#     train_accuracy_list = []
#     test_accuracy_list = []
#     roc_auc_list = []
#     fprs = []
#     tprs = []
#     precisions = []
#     recalls = []
#     average_precision_list = []
#     agg_cm = np.zeros((20,20),dtype=int)
#     log_reg_auc_list = []
#     log_reg_accuracy_list = []
#     log_reg_conf_matrices = []
#     score_bins = np.arange(0, 101, 10)  # 0 to 100 in steps of 10
#
#     # Initialize lists to store categorized CNC scores
#     bin_categories_rf = {'True Positive': [], 'True Negative': [], 'False Positive': [], 'False Negative': []}
#     bin_categories_lr = {'True Positive': [], 'True Negative': [], 'False Positive': [], 'False Negative': []}
#
#
#     for train_index, test_index in kf.split(features):
#         X = features
#         y = target
#         X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
#         y_train = y.iloc[train_index]
#         y_test = y.iloc[test_index]
#         combined_train = pd.concat([X_train, y_train], axis=1) #combining is required to run CreateCombinedDataset
#         combined_test = pd.concat([X_test, y_test], axis=1)
#         sided_train = CreateCombinedDataset(combined_train) #also runs binning operation
#         sided_test = CreateCombinedDataset(combined_test)
#
#         #Sidedness operations
#         X_train = sided_train.iloc[:, :-1]  #-1 refers to the column "CNC_Bins" and is for ordinal classification. Change to -2 for CNC score
#         y_train = sided_train.iloc[:, -1]
#         X_test = sided_test.iloc[:, :-1]
#         y_test = sided_test.iloc[:, -1]
#
#         # training the classifier on ordinal, 20 bins
#         y_test_class = y_test
#         clf = RandomForestClassifier(n_estimators=100, random_state=42)
#         clf.fit(X_train, y_train)
#         y_pred_test_class = clf.predict(X_test)
#
#         cm = confusion_matrix(y_test, y_pred_test_class)
#         agg_cm += cm
#
#         # Predict probabilities for the test set
#         y_test_probabilities = clf.predict_proba(X_test) # Probability of CNC < 40
#
#         # print(f"Fold Number: {y_test_probabilities.shape}")
#         y_test_probabilities_percentage = y_test_probabilities * 100  # Convert to percentage
#
#
#
#         # Evaluate the classifier
#         roc_auc = roc_auc_score(y_test_class, y_test_probabilities, multi_class='ovr') # One vs one
#         roc_auc_list.append(roc_auc)
#         #computing false and true rates
#         fpr, tpr, _ = roc_curve(y_test_class, y_test_probabilities)
#         fprs.append(fpr)
#         tprs.append(tpr)
#
#         ##necessary variables for precision-recall curve
#         # Compute precision and recall
#         precision, recall, _ = precision_recall_curve(y_test_class, y_test_probabilities)
#         precisions.append(precision)
#         recalls.append(recall)
#
#         # Compute average precision for this fold
#         avg_precision = average_precision_score(y_test_class, y_test_probabilities)
#         average_precision_list.append(avg_precision)
#
#
#         ##Running logistic regression comparison
#         log_reg = LogisticRegression(random_state=42, max_iter=1000)
#         log_reg.fit(X_train, y_train_class)
#
#         # Predict probabilities and classes
#         y_test_probabilities = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1
#         y_test_pred_class_log = log_reg.predict(X_test)  # Predicted classes
#
#         # Calculate metrics
#         auc = roc_auc_score(y_test_class, y_test_probabilities)
#         accuracy = accuracy_score(y_test_class, y_test_pred_class_log)
#         conf_matrix = confusion_matrix(y_test_class, y_test_pred_class_log)
#
#         # Store results
#         log_reg_auc_list.append(auc)
#         log_reg_accuracy_list.append(accuracy)
#         log_reg_conf_matrices.append(conf_matrix)
#         #####Histogram set-up
#         # Classify each data point for Random Forest
#         rf_classification = np.select(
#             [
#                 (y_test_class == 1) & (y_pred_test_class == 1),  # TP
#                 (y_test_class == 0) & (y_pred_test_class == 0),  # TN
#                 (y_test_class == 0) & (y_pred_test_class == 1),  # FP
#                 (y_test_class == 1) & (y_pred_test_class == 0),  # FN
#             ],
#             ['True Positive', 'True Negative', 'False Positive', 'False Negative']
#         )
#
#         # Classify each data point for Logistic Regression
#         lr_classification = np.select(
#             [
#                 (y_test_class == 1) & (y_test_pred_class_log == 1),  # TP
#                 (y_test_class == 0) & (y_test_pred_class_log == 0),  # TN
#                 (y_test_class == 0) & (y_test_pred_class_log == 1),  # FP
#                 (y_test_class == 1) & (y_test_pred_class_log == 0),  # FN
#             ],
#             ['True Positive', 'True Negative', 'False Positive', 'False Negative']
#         )
#
#         # Append CNC scores to respective classification bins for Random Forest
#         for cnc_score, classification in zip(y_test, rf_classification):
#             bin_categories_rf[classification].append(cnc_score)
#
#         # Append CNC scores to respective classification bins for Logistic Regression
#         for cnc_score, classification in zip(y_test, lr_classification):
#             bin_categories_lr[classification].append(cnc_score)
#
#     print("K-Fold MSE Scores:", mse_scores)
#     print("Average MSE:", np.mean(mse_scores))
#     print("Test Accuracy:", test_accuracy_list)
#     print("Average Test Accuracy",np.mean(test_accuracy_list))
#     print("ROC AUC:", roc_auc_list)
#
# #########Visualizations
#
# ###ROC curves
# plt.figure(figsize=(10, 6))
# for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
#     roc_auc = AUC(fpr, tpr)  # Calculate the area under the curve
#     plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AU-ROC = {roc_auc:.2f})')
#
# # Plot the diagonal (random chance)
# plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=2, label='Random Chance')
#
# # Average AU-ROC
# mean_fpr = np.linspace(0, 1, 100)
# mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
# mean_auc = AUC(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, color='red', linestyle='-', lw=2.5, label=f'Mean ROC (AU-ROC = {mean_auc:.2f})')
#
# # Add labels, title, and legend
# plt.title('ROC Curves for K-Fold Cross-Validation', fontsize=16)
# plt.xlabel('False Positive Rate (FPR)', fontsize=14)
# plt.ylabel('True Positive Rate (TPR)', fontsize=14)
# plt.legend(loc='lower right', fontsize=12)
#
# # Show the plot
# plt.tight_layout()
# plt.show()
#
#
# # Plot the Precision-Recall curve for each fold
# plt.figure(figsize=(10, 6))
# for i, (precision, recall) in enumerate(zip(precisions, recalls)):
#     avg_precision = average_precision_list[i]  # Retrieve the average precision for the fold
#     plt.plot(recall, precision, lw=2, label=f'Fold {i+1} (AP = {avg_precision:.2f})')
#
# # Plot the mean Precision-Recall curve
# mean_precision = np.mean([np.interp(np.linspace(0, 1, 100), recall, precision) for recall, precision in zip(recalls, precisions)], axis=0)
# mean_recall = np.linspace(0, 1, 100)
# mean_avg_precision = np.mean(average_precision_list)
# plt.plot(mean_recall, mean_precision, color='red', linestyle='-', lw=2.5, label=f'Mean Precision-Recall Curve (AP = {mean_avg_precision:.2f})')
#
# # Add labels, title, and legend
# plt.title('Precision-Recall Curves for K-Fold Cross-Validation', fontsize=16)
# plt.xlabel('Recall', fontsize=14)
# plt.ylabel('Precision', fontsize=14)
# plt.legend(loc='lower left', fontsize=12)
#
# # Show the plot
# plt.tight_layout()
# plt.show()
#
#
# ##Confusion matrix plot
# disp = ConfusionMatrixDisplay(confusion_matrix=cm / cm.sum(axis=1, keepdims=True))
#                               #,display_labels=['CNC > 40', 'CNC ≤ 40'])
# disp.plot(cmap='Blues', values_format='.2f')  # Format as decimals
#
# plt.title(f'Confusion Matrix Aggregated Across Folds')
# plt.show()
# #
# #
# # ######LOG REGRESSION CALCS
# # # Calculate mean and standard deviation for AUC and accuracy
# # mean_auc = np.mean(log_reg_auc_list)
# # std_auc = np.std(log_reg_auc_list)
# # mean_accuracy = np.mean(log_reg_accuracy_list)
# # std_accuracy = np.std(log_reg_accuracy_list)
# #
# # print(f"Logistic Regression Mean AUC: {mean_auc:.2f} ± {std_auc:.2f}")
# # print(f"Logistic Regression Mean Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(1, kf.get_n_splits() + 1), roc_auc_list, marker='o', label='Random Forest AU-ROC')
# # plt.plot(range(1, kf.get_n_splits() + 1), log_reg_auc_list, marker='o', label='Logistic Regression AU-ROC')
# # plt.xlabel('Fold')
# # plt.ylabel('AU-ROC Score')
# # plt.title('AU-ROC Comparison Across Folds')
# # plt.legend()
# # plt.grid()
# # plt.show()
# #
# #
#
# ####Histogram
# # Convert categorized scores to arrays for histogram stacking
# # Convert categorized scores to arrays for histogram stacking
# def prepare_histogram_data(bin_categories):
#     hist_data = []
#     for classification in ['True Positive', 'True Negative', 'False Positive', 'False Negative']:
#         hist_data.append(bin_categories[classification])
#     return hist_data
#
# # Use a seaborn color palette for better colors
# colors = sns.color_palette("pastel", 4)  # Choose a soft pastel color palette
#
# # Prepare data for Random Forest and Logistic Regression
# rf_hist_data = prepare_histogram_data(bin_categories_rf)
# lr_hist_data = prepare_histogram_data(bin_categories_lr)
#
# # Plotting histograms
# fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
#
# # Random Forest Histogram
# axes[0].hist(
#     rf_hist_data, bins=score_bins, stacked=True, color=colors, alpha=0.9, edgecolor='black'
# )
# axes[0].set_title('Random Forest Predictions by CNC Score')
# axes[0].set_xlabel('CNC Score')
# axes[0].set_ylabel('Count')
# axes[0].legend(['True Positive', 'True Negative', 'False Positive', 'False Negative'], loc='upper right')
# axes[0].grid(axis='y', linestyle='--', alpha=0.7)
#
# # Logistic Regression Histogram
# axes[1].hist(
#     lr_hist_data, bins=score_bins, stacked=True, color=colors, alpha=0.9, edgecolor='black'
# )
# axes[1].set_title('Logistic Regression Predictions by CNC Score')
# axes[1].set_xlabel('CNC Score')
# axes[1].legend(['True Positive', 'True Negative', 'False Positive', 'False Negative'], loc='upper right')
# axes[1].grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.show()
#
# ####Calibration curve
# # Function to plot calibration curve
# def plot_calibration_curve(model_name, y_true, y_probs, ax):
#     # Compute calibration curve
#     fraction_of_positives, mean_predicted_value = calibration_curve(
#         y_true, y_probs, n_bins=10, strategy='uniform'
#     )
#     ax.plot(mean_predicted_value, fraction_of_positives, marker='o', label=model_name)
#     ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
#     ax.set_xlabel('Mean Predicted Probability')
#     ax.set_ylabel('Fraction of Positives')
#     ax.set_title(f'Calibration Curve for {model_name}')
#     ax.legend()
# #
# # Plotting calibration curves for both models
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#
# # Random Forest Calibration Curve
# y_test_class_rf = (y_test <= 40).astype(int)  # Binary target for Random Forest
# rf_probs = clf.predict_proba(X_test)[:, 1]  # Probability of CNC < 40
# plot_calibration_curve('Random Forest', y_test_class_rf, rf_probs, axes[0])
#
# # Logistic Regression Calibration Curve
# log_reg_probs = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1
# plot_calibration_curve('Logistic Regression', y_test_class_rf, log_reg_probs, axes[1])
#
# plt.tight_layout()
# plt.show()
# ###############################################
#     #Single model no k-fold
#     # best_model.fit(X_train_scaled,y_train)
#     # y_pred_train = best_model.predict(X_train_scaled)
#     # y_pred_test = best_model.predict(X_test_scaled)
#     # mse_score = mean_squared_error(y_test,y_pred_test)
#     # print(f'MSE Score: {mse_score}')
#
#
#     # # binary targets for CNC less than 40% classification (1 if <= 40, else 0)
#     # y_train_class = (y_train <= 40).astype(int)  # Binary target for training
#     # y_test_class = (y_test <= 40).astype(int)  # Binary target for testing
#     # y_pred_train_class = (y_pred_train <=40).astype(int)
#     # y_pred_test_class = (y_pred_test <=40).astype(int)
#     #
#     # # Calculate accuracy for the training set
#     # train_accuracy = accuracy_score(y_train_class, y_pred_train_class)
#     # print(f'Training Accuracy at Predicting CNC < 40: {train_accuracy:.4f}')
#     #
#     # # Calculate accuracy for the test set
#     # test_accuracy = accuracy_score(y_test_class, y_pred_test_class)
#     # print(f'Test Accuracy at Predicting CNC < 40: {test_accuracy:.4f}')
#     #
#     # # Train a RandomForestClassifier for probability predictions
#     # clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     # clf.fit(X_train_scaled, y_train_class)
#     #
#     # # Predict probabilities for the test set
#     # y_test_probabilities = clf.predict_proba(X_test_scaled)[:, 1]  # Probability of CNC < 40
#     # y_test_probabilities_percentage = y_test_probabilities * 100  # Convert to percentage
#     #
#     # # Evaluate the classifier
#     # roc_auc = roc_auc_score(y_test_class, y_test_probabilities)
#     # print(f"ROC-AUC Score: {roc_auc:.4f}")
#     #
#     # # Display percent risk scores
#     # for i, (true_score, probability) in enumerate(zip(y_test, y_test_probabilities_percentage)):
#     #     print(f"User {i + 1}: True CNC Score = {true_score:.2f}, Risk of CNC < 40 = {probability:.2f}%")
#     #
#     #
#     # # Calculate the ROC-AUC score
#     # roc_auc = roc_auc_score(y_test_class, y_test_probabilities)
#     # print(f"ROC-AUC Score for Probability: {roc_auc:.4f}")
#     #
#     # # Plot the ROC curve
#     # fpr, tpr, thresholds = roc_curve(y_test_class, y_test_probabilities)
#     # plt.figure(figsize=(8, 6))
#     # plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
#     # plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
#     # plt.xlabel('False Positive Rate')
#     # plt.ylabel('True Positive Rate')
#     # plt.title('ROC Curve')
#     # plt.legend(loc='best')
#     # plt.grid()
#     # plt.show()
#     #
#     # cm = confusion_matrix(y_test_class, y_pred_test_class)
#     # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['>= 0.4', '< 0.4'])
#     # disp.plot(cmap=plt.cm.Blues)
#     # plt.title("Confusion Matrix")
#     # plt.show()
#     #
#     #
#     # # Calculate Precision-Recall curve
#     # precision, recall, _ = precision_recall_curve(y_test_class, y_pred_test_class)
#     # pr_auc = auc(recall, precision)  # Area Under the Precision-Recall Curve
#     #
#     # # Set the aesthetic style of the plots
#     # sns.set(style="whitegrid")
#     #
#     # # Plotting Precision-Recall curve with enhancements
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(recall, precision, marker='o', color='b', label=f'Precision-Recall curve (AUC = {pr_auc:.2f})', lw=2)
#     # plt.fill_between(recall, precision, alpha=0.1, color='blue')
#     # plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
#     # plt.xlabel('Recall', fontsize=14)
#     # plt.ylabel('Precision', fontsize=14)
#     # plt.xlim([0.0, 1.0])
#     # plt.ylim([0.0, 1.0])
#     # plt.grid(True, linestyle='--', alpha=0.7)
#     # plt.axhline(0.5, linestyle='--', color='gray', label='Baseline Precision = 0.5')  # Baseline for comparison
#     # plt.axvline(0.5, linestyle='--', color='gray')  # Baseline for recall
#     # plt.legend(loc="lower left", fontsize=12)
#     # plt.tight_layout()  # Adjust layout to fit everything nicely
#     # plt.show()
#     #
#     # # Print AUC
#     # print(f'Area Under Precision-Recall Curve: {pr_auc:.4f}')
#     #
#     #
#     # ##Histogram
#     #
#     #
#     # # Set the aesthetic style of the plots
#     # sns.set(style="whitegrid")
#     #
#     # # Create a histogram of CNC scores
#     # plt.figure(figsize=(10, 6))
#     # sns.histplot(target, bins=20, kde=True, color='blue', alpha=0.6)  # KDE adds a density estimate line
#     # plt.title('Distribution of CNC Scores', fontsize=16, fontweight='bold')
#     # plt.xlabel('CNC Scores', fontsize=14)
#     # plt.ylabel('Frequency', fontsize=14)
#     # plt.axvline(x=40, color='red', linestyle='--', label='Threshold (40)')  # Optional: Add a threshold line
#     # plt.legend()
#     # plt.grid(True)
#     # plt.tight_layout()  # Adjust layout to fit everything nicely
#     # plt.show()
#
#
#