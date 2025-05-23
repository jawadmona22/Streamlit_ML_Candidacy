import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold,GridSearchCV,GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

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
    percentiles = np.arange(0,100,11)
    thresholds = np.unique(np.percentile(data['CNC'],percentiles))
    print(f"Thresholds {thresholds}")
    data['CNC_bin'] = np.digitize(data['CNC'], thresholds, right=True)
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


def preprocess_data(file_name,sampling):
    '''The purpose of this section is to create the dataset that will be used to
    feed in to the different types of models. The dataset will be split into features/targets
    and train/test and will be converted into ordinal.

    Parameters will be set so that we can control from the beginning whether or not
    we want to have SMOTE/Oversampling/Neither present and so that we can have
    CNC binned or not binned at all.'''

    raw_data = pd.read_csv(file_name)
    columns_of_interest = [  # Final string specifies target, all others are feature columns
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
    ]

    # Split into features and target, but does not separate by ear to preserve patient ID
    features, target = SetFeaturesAndTarget(raw_data,
                                            columns_of_interest)  # target includes CNC_L and CNC_R, features include both
    full_dataset = pd.concat([features, target], axis=1)
    ##Add patient ID
    full_dataset['patient_id'] = range(1, len(full_dataset) + 1)


    # Apply SMOTE
    if sampling=='SMOTE':
        combined_sides_data, bin_threshold = CreateCombinedDataset(full_dataset, num_bins=10, plot=True)
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
        #Equal *spaced* bins with SMOTE applied
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        # Extend `groups_train` to match SMOTE's new sample count
        num_new_samples = len(X_train_resampled) - len(X_train)
        # Assign synthetic samples a placeholder group (-1)
        groups_resampled = np.concatenate([groups_train, np.full(num_new_samples, -1)])
        X_train, y_train, groups_train = X_train_resampled, y_train_resampled, groups_resampled

        print("Class distribution after SMOTE:", np.bincount(y_train))

    elif sampling=='None':
        #No binning, just pure CNC values 1-100
        print("No oversampling applied")

    elif sampling=='Percentiles':
        print("Data divided into percentiles")
        combined_sides_data, bin_threshold = CreateCombinedDataset(full_dataset, num_bins=10, plot=True)
        X = combined_sides_data.drop(columns=['CNC_bin', 'CNC', 'patient_id'])
        y = combined_sides_data['CNC_bin']
        groups = combined_sides_data['patient_id'].values
        # Train-test split
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups[train_idx]  # Extract corresponding train groups


    elif sampling == 'Oversampling':
        print(sampling)
        #Equal *spaced* bins with oversampling applied


    return X_train,y_train,groups_train,X_test,y_test,bin_threshold


def train_model(X_train,y_train,groups_train):
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
            'classifier': [XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 6, 10],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    ]

    # Perform GridSearchCV with GroupKFold
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='f1_micro',
        cv=kf,
        n_jobs=-1
    )

    print(f"X train: {X_train.columns}")
    print(f"y Train: {y_train}")



    grid_search.fit(X_train, y_train, groups=groups_train)
    with open('grid_search_results.pkl', 'wb') as f:
        pickle.dump(grid_search, f)
    return grid_search



#Calibration Curves with histogram

#Table Comparing Model Peformance








def run_visualizations(grid_search,X_test,y_test,bin_threshold):
    best_models = {}

    # Iterate through all models tested in GridSearchCV
    for i, param in enumerate(grid_search.cv_results_['params']):
        model_name = param['classifier'].__class__.__name__
        mean_score = grid_search.cv_results_['mean_test_score'][i]  # Extract score for this model

        if model_name not in best_models or mean_score > best_models[model_name]['score']:
            best_models[model_name] = {
                'model': grid_search.best_estimator_.named_steps['classifier'],  # Extract the classifier itself
                'score': mean_score
            }


    for model_name, model_info in best_models.items():
        best_model = model_info['model']
        y_true_binary = (y_test <= bin_threshold).astype(int)

        # Get predicted probabilities for all classes
        y_pred_prob = best_model.predict_proba(X_test)

        # Get the class labels
        class_labels = np.unique(y_test)

        # Identify which classes are below the bin_threshold
        classes_below_threshold = class_labels[class_labels <= bin_threshold]

        # Sum the predicted probabilities for these classes
        # For each sample, sum the probabilities of classes below the threshold
        y_pred_prob_binary = np.sum(y_pred_prob[:, classes_below_threshold], axis=1)
        print(y_pred_prob_binary)
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob_binary)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"Model (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC AUC Curve (Threshold = {bin_threshold})")
        plt.legend()
        plt.grid()


        plt.show()

        print(f"AUC for binary classification based on threshold {bin_threshold}: {roc_auc:.3f}")

###MAIN CALL###
def operator(params):
    file_name = params["file_name"]
    for sampling in params["oversampling"]:
        X_train, y_train, groups_train, X_test, y_test,bin_threshold = preprocess_data(file_name=file_name, sampling=sampling)
        if params['train'] == True:
            grid_search = train_model(X_train, y_train, groups_train, X_test, y_test)
            best_model = grid_search.best_estimator_  # best model
            with open(f'grid_search_results_{sampling}.pkl', 'wb') as f:
                pickle.dump(grid_search, f)
        else:
            with open('grid_search_results_Percentiles.pkl', 'rb') as f:
                grid_search = pickle.load(f)

        run_visualizations(grid_search, X_test, y_test,bin_threshold)


params = {
    'file_name': 'candidacy_v3.csv',
    'oversampling':['SMOTE','Percentiles'],
    'train':False
}
operator(params)



