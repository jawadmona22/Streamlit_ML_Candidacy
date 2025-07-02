import pandas as pd
from sklearn.model_selection import GroupKFold,GridSearchCV,GroupShuffleSplit,ParameterGrid,KFold
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
from pathlib import Path
from scipy.stats import t
from sklearn import preprocessing
from ordinal import OrdinalClassifier

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




def sixty_sixty_predictions(X_test):
    y_pred = []

    for index, row in X_test.iterrows():
        hz_500 = row["hz500"]
        hz_1000 = row["hz1000"]
        hz_2000 = row["hz2000"]
        wrs = row["WRS"]

        PTA = (hz_500 + hz_1000 + hz_2000)/3

        if PTA >= 60 and wrs < 60:
            y_pred.append(1)  # They are a candidate
        else:
            y_pred.append(0)

    return y_pred
def sixty_sixty_predictions_az_bio(X_test):
    y_pred = []
    for index,row in X_test.iterrows():
        #Determine better ear
        hz_500_L = row["hz500_L"]
        hz_1000_L = row["hz1000_L"]
        hz_2000_L = row["hz2000_L"]
        wrs_L= row["WRS_L"]

        hz_500_R = row["hz500_R"]
        hz_1000_R = row["hz1000_R"]
        hz_2000_R= row["hz2000_R"]
        wrs_R= row["WRS_R"]


        if wrs_L  >= wrs_R: #Case: left is better ear
            PTA = (hz_500_L + hz_1000_L + hz_2000_L) / 3
            if PTA >= 60 and wrs_L < 60:
                y_pred.append(1)  # They are a candidate
            else:
                y_pred.append(0)

        if wrs_R > wrs_L: #Case: left is better ear
            PTA = (hz_500_R + hz_1000_R + hz_2000_R) / 3
            if PTA >= 60 and wrs_R < 60:
                y_pred.append(1)  # They are a candidate
            else:
                y_pred.append(0)








    return y_pred



def sixty_sixty_run():
    all_labels = [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
    ]
    filtered_dataset = Clean_Data(debug=False, all_labels=all_labels)
    left_right_data = Create_Left_Right_Data(filtered_dataset, debug=False)
    X_test = left_right_data.drop(columns=[ 'CNC', 'patient_id'])
    y_test = left_right_data['CNC']

    # Get predictions
    y_pred = sixty_sixty_predictions(X_test)
    y_true_binary = (y_test < 50).astype(int)

    # Accuracy
    accuracy = accuracy_score(y_true_binary, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true_binary, y_pred)
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
    ax.set_title(f"60/60 Rule", fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'TRIO-figs/60-60-cm-cnc.png')

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # same as recall
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    print(
        f"Precision: {precision}, Recall: {recall}, F1: {f1}, Sensitivity:{sensitivity}, Specificity: {specificity}")

def sixty_sixty_run_az_bio():
    all_labels = [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'AzBioQuiet_bi'
    ]
    filtered_dataset = Clean_Data(debug=False, all_labels=all_labels)
    X_test = filtered_dataset.drop(columns=['AzBioQuiet_bi'])
    y_test = filtered_dataset['AzBioQuiet_bi']

    # Get predictions
    y_pred = sixty_sixty_predictions_az_bio(X_test)
    y_true_binary = (y_test < 60).astype(int)
    print("Shape Y Pred:", len(y_pred))
    print("Shape Y True:", len(y_true_binary))
    # Accuracy
    accuracy = accuracy_score(y_true_binary, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true_binary, y_pred)
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
    ax.set_title(f"60/60 Rule", fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'TRIO-figs/60-60-cm-azbio.png')

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # same as recall
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    print(
        f"Precision: {precision}, Recall: {recall}, F1: {f1}, Sensitivity:{sensitivity}, Specificity: {specificity}")



def Post_Iter_Processing(folder_name = ''):
    # Load the combined predictions
    if folder_name != '':
        folder = Path(f'{folder_name}')
    else:
        folder = Path('bins-pkls')
    iter_info = []

    #Set up plotting for later for AUC so that the path issue from earlier doesn't happen
    auc_fig, auc_ax = plt.subplots(figsize=(8, 6))
    root_folder = folder_name.split('/')[0]

    for pkl_file in folder.glob('*.pkl'):  ##Essentially loops over models because they're each in a different .pkl
        print(f"Loading: {pkl_file.name}")
        df = pd.read_pickle(f"{folder_name}/{pkl_file.name}")
        ModelName = df['ModelName'][0]
        if ModelName == 'XGClassifer':
            ModelName = 'XGBoostClassifier'

        suffix = pkl_file.name.split('_')[2].split('.')[0]
        if suffix == 'reg':
                tag = "No SMOTE"
        else:
            # tag = suffix.upper() #SMOTE is the other option
            tag = "SMOTE"




        # --- Aggregate across folds ---
        summary = df.groupby(['ModelName']).agg(
            f1_mean=('F1', 'mean'),
            f1_std=('F1', 'std'),
            auc_mean=('AUC', 'mean'),
            auc_std=('AUC', 'std'),
            n_folds=('AUC', 'count'),
            sensitivity=('Sensitivity','mean'),
            specificity=('Specificity','mean'),
            precision=('Precision','mean'),
            recall=('Recall','mean'),
        ).reset_index()

        summary['f1_95ci'] = t.ppf(0.975, summary['n_folds'] - 1) * summary['f1_std'] / np.sqrt(summary['n_folds'])
        summary['auc_95ci'] = t.ppf(0.975, summary['n_folds'] - 1) * summary['auc_std'] / np.sqrt(summary['n_folds'])

        summary_sorted = summary.sort_values('auc_mean', ascending=False)

        first_row = summary_sorted[[
            'ModelName',
            'f1_mean', 'f1_std', 'f1_95ci',
            'auc_mean', 'auc_std', 'auc_95ci',
            'sensitivity', 'specificity', 'precision', 'recall'
        ]].iloc[:3]

        first_row.round(3)
        iter_info.append(summary_sorted.copy())
        print("\n Best Model Summary (Top Row):")
        print(first_row.to_string())

        # ---- Compute Confusion Matrix: All Folds, Each Model ---- #


        totals = df[['tp', 'tn', 'fp', 'fn']].sum()
        tp, tn, fp, fn = totals['tp'], totals['tn'], totals['fp'], totals['fn']

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
        fig_1, ax = plt.subplots(figsize=(6, 4))

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
        plt.savefig(f'{folder_name}/{ModelName}{suffix}-{root_folder}-cm.png')
        plt.close(fig_1)

        # ---- Compute Calibration Curve: All Folds, Each Model ---- #
        all_prob_trues = []
        all_prob_preds = []
        n_bins = 10
        all_probs = []
        for _, row in df.iterrows():
            y_true = np.array(row['y_true_binary'])

            if 'binary' in root_folder:
                y_score = np.array(row['y_pred_prob_binary'])

                y_score_binary = y_score
            else:
                y_score = np.array(row['y_pred_proba'])
                y_score_binary = np.clip(y_score[:, :7].sum(axis=1), 0, 1)

            prob_true, prob_pred = calibration_curve(y_true, y_score_binary, n_bins=10, strategy='uniform')
            all_prob_trues.append(prob_true)
            all_prob_preds.append(prob_pred)
            all_probs.extend(y_score_binary)

        #Pad arrays to consistent length
        def pad(arr, length):
            padded = np.full(length, np.nan)
            padded[:len(arr)] = arr
            return padded

        all_prob_trues = np.array([pad(arr, n_bins) for arr in all_prob_trues])
        all_prob_preds = np.array([pad(arr, n_bins) for arr in all_prob_preds])

        mean_true = np.nanmean(all_prob_trues, axis=0)
        mean_pred = np.nanmean(all_prob_preds, axis=0)
        fig, ax = plt.subplots(figsize=(8, 6))

        counts, bins = np.histogram(all_probs, bins=10, range=(0, 1))
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

        ax.plot(mean_pred, mean_true, 'o-')

        ax.plot([0, 1], [0, 1], 'k--')

        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{ModelName} {tag}")
        plt.tight_layout()
        plt.savefig(f"TRIO-figs/{ModelName}-{tag}-{root_folder}-cb.png")
        # plt.show()




        # ---- Compute AUC ---- #
        aucs = []
        tprs = []
        mean_fpr = np.linspace(0,1,100)
        for _, row in df.iterrows():
            y_true = np.array(row['y_true_binary'])

            if 'binary' in root_folder:
                y_score = np.array(row['y_pred_prob_binary'])

                y_score_binary = y_score
            else:
                y_score = np.array(row['y_pred_proba'])
                y_score_binary = np.clip(y_score[:, :7].sum(axis=1), 0, 1)

            fpr, tpr, thresholds = roc_curve(y_true, y_score_binary)
            roc_auc = auc(fpr, tpr)

            #Interpolate according to Scikit website: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

            interp_tpr = np.interp(mean_fpr,fpr,tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)





        mean_tpr = np.mean(tprs,axis=0)
        mean_tpr[-1] = 1.0
        std_auc = np.std(aucs)
        mean_auc = auc(mean_fpr, mean_tpr)


        auc_ax.plot(
            mean_fpr,
            mean_tpr,
            label=rf"{ModelName} ({tag}) (Mean AUC = {mean_auc:.2f} ± {std_auc:.3f})",
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        auc_ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            alpha=0.2,
           )

        auc_ax.set_xlabel("False Positive Rate")
        auc_ax.set_ylabel("True Positive Rate")
        auc_ax.set_title(f"Mean ROC ({tag})")

        auc_ax.legend(
            loc="lower right",
            fontsize="small",
            frameon=True,
            handlelength=2.5,
            borderpad=0.5,
        )

        auc_ax.grid(False)

    auc_fig.tight_layout()
    auc_fig.savefig(f"TRIO-figs/AUROC-{root_folder}-{tag}")

    #----- Histograms of SMOTE Distributions ---- #
    deltas = []
    for _, row in df.iterrows():
        ##SMOTE procedure
        smote_before = np.array(row['SMOTE_before'])
        smote_after = np.array(row['SMOTE_after'])
        delta = smote_after - smote_before
        deltas.append(delta)

    delta_sums = np.sum(deltas, axis=0)[0]
    delta_means = np.mean(deltas,axis=0)[0]
    plt.close('all')
    plt.figure()
    print(delta_sums)
    plt.bar(range(11), delta_sums)

    plt.xlabel('Bin')
    plt.ylabel('Sum of Changes Across Inner K-Fold')
    plt.title('SMOTE Delta Changes')
    plt.show()

    plt.close('all')
    plt.figure()
    print(delta_means)
    plt.bar(range(11), delta_means)

    plt.xlabel('Bin')
    plt.ylabel('Mean Changes Across Inner K-Fold')
    plt.title('SMOTE Average Delta')
    plt.show()

    summary_df = pd.concat(iter_info, ignore_index=True)
    print(summary_df.head())
    summary_df.to_excel(f"TRIO-figs/Summary_Iter-{tag}-{root_folder}.xlsx")




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



class DebugSMOTE(SMOTE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.before_counts_ = []
        self.after_counts_ = []

    def fit_resample(self, X, y):
        self.before_counts_.append(np.bincount(y))
        X_res, y_res = super().fit_resample(X, y)
        self.after_counts_.append(np.bincount(y_res))
        return X_res, y_res

def nested_cross_optimize_ear_specific(n_iters=1, k_outer=10, k_inner=10, all_labels=[], raw_preds=False, smote=False, file_name='', folder_name='', target='CNC'):


    if smote:
        pipeline = Pipeline([
        ('smote', DebugSMOTE(random_state=42)),
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
            {
                'classifier': [RandomForestClassifier(random_state=42)],
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [2, 3, 4, 5, None],
                'classifier__min_samples_split': [2, 4, 8],
                'classifier__min_samples_leaf': [1, 2, 4]
            }],
        'XGClassifer':
        [{  # XGBoost
            'classifier': [XGBClassifier(eval_metric='logloss', random_state=42)],
            'classifier__n_estimators': [150, 300, 450],
            'classifier__max_depth': [2,4,6],
            'classifier__reg_alpha': [0, 0.01,.5]

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
        X = left_right_data.drop(columns=[f'{target}','patient_id'])
        y = left_right_data[f'{target}']
        y = (y < 50).astype(int)
        groups = left_right_data['patient_id'].values



    for model_name, param_subset in param_grid.items():
        print(f"Running procedure for {model_name}")
        # base_model = param_subset[0]["classifier"][0]
        # ordinal_clf = OrdinalClassifier(estimator=base_model)
        # pipeline.set_params(classifier=ordinal_clf)

        iter_results = []
        for iter in tqdm(range(0,n_iters)):

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
                print(f"Iteration {iter}, fold {fold}")

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                groups_train = groups[train_idx]

                #Inner loop GridSearch


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

                smote_step = best_model.named_steps.get('smote', None)

                if smote_step is not None and hasattr(smote_step, 'before_counts_'):
                    smote_before = smote_step.before_counts_
                    smote_after = smote_step.after_counts_
                else:
                    smote_before = None
                    smote_after = None
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
                    pred_prob_below_thresh = y_pred_prob_binary

                #Compute Metrics for the fold
                fpr, tpr, _ = roc_curve(y_true_binary, pred_prob_below_thresh)
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
                    'y_pred_proba': y_pred_prob,
                    'y_scores':pred_prob_below_thresh,
                    'y_true':y_test,
                    'y_true_binary': y_true_binary,
                    'y_pred_binary': y_pred_binary,
                    'y_pred_prob_binary':y_pred_prob_binary,
                    'SMOTE_before': smote_before,
                    'SMOTE_after': smote_after

                }
                iter_results.append(new_row)



        all_iter_results = pd.DataFrame(iter_results)

        all_iter_results.to_pickle(f'{folder_name}/{model_name}_nested_{file_name}.pkl')

        print(all_iter_results.head())

        print(f"Shape of final dataframe: {all_iter_results.shape}")
def nested_cross_optimize_bilateral(n_iters=1, k_outer=10, k_inner=10, all_labels=[], smote=False, file_name='', folder_name='', target='AzBioQuiet_bi'):

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
            'classifier__learning_rate': [0.05, 0.01],
            'classifier__n_estimators': [150,300,450],
            'classifier__subsample':[0.5,.7],

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
    X = filtered_dataset.drop(columns=[f'{target}','patient_id'])
    y = filtered_dataset[f'{target}']
    y = (y < 60).astype(int)   #60% for AzBio
    groups = filtered_dataset['patient_id'].values



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



                y_true_binary = y_test
                y_pred_binary = y_pred
                y_pred_prob_binary = y_pred_prob[:, 1]
                pred_prob_below_thresh = y_pred_prob_binary

                #Compute Metrics for the fold
                fpr, tpr, _ = roc_curve(y_true_binary, pred_prob_below_thresh)
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
                    'y_pred_proba': y_pred_prob,
                    'y_scores':pred_prob_below_thresh,
                    'y_true':y_test,
                    'y_true_binary': y_true_binary,
                    'y_pred_binary': y_pred_binary,
                    'y_pred_prob_binary':y_pred_prob_binary

                }
                iter_results.append(new_row)



        all_iter_results = pd.DataFrame(iter_results)

        all_iter_results.to_pickle(f'{folder_name}/{model_name}_nested_{file_name}.pkl')

        print(all_iter_results.head())

        print(f"Shape of final dataframe: {all_iter_results.shape}")



def key_set_optimization():
    all_labels =  [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
    ]

    # nested_cross_optimize_ear_specific(n_iters=1, k_outer=2, k_inner=2, all_labels=all_labels, raw_preds=False, smote=True, file_name='test-ord', folder_name='bins-pkls')
    # nested_cross_optimize_ear_specific(n_iters=100, k_outer=10, k_inner=10, all_labels=all_labels, raw_preds=False, smote=False, file_name='reg', folder_name='bins-pkls')
    # nested_cross_optimize_ear_specific(n_iters=100, k_outer=10, k_inner=10, all_labels=all_labels, raw_preds=False, smote=True, file_name='upg', folder_name='bins-pkls') #upgraded parameters
    # nested_cross_optimize_ear_specific(n_iters=100, k_outer=10, k_inner=10, all_labels=all_labels, raw_preds=True, smote=False, file_name='reg', folder_name='binary-pkls')
    # nested_cross_optimize_ear_specific(n_iters=100, k_outer=10, k_inner=10, all_labels=all_labels, raw_preds=True, smote=True, file_name='smote', folder_name='binary-pkls')
    # Post_Iter_Processing(folder_name='bins-pkls/SMOTE')
    # Post_Iter_Processing(folder_name='bins-pkls/No-SMOTE')
    # Post_Iter_Processing(folder_name='binary-pkls/SMOTE')
    # Post_Iter_Processing(folder_name='binary-pkls/No-Smote')
    # Post_Iter_Processing(folder_name='bins-pkls/upg')
    # Post_Iter_Processing(folder_name='bins-pkls/ordinal')
    Post_Iter_Processing(folder_name='bins-pkls/test')
def feature_set_optimize():
    set_dict = {
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
        "wrs-a":
        #
        # Audiogram + WRS + Age
            [
                'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
                'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
                'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
                'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
                'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
                'WRS_L', 'WRS_R', 'Age', 'CNC_L', 'CNC_R'
            ],
        # "wrs-a-dhl":

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

    # for key, value in set_dict.items():
    #     nested_cross_optimize(n_iters=1, k_outer=10, k_inner=10, all_labels=value,raw_preds=False,smote=False,file_name=key,folder_name='feature-selection-pkls')
    for key, value in set_dict.items():
        Post_Iter_Processing(folder_name=f'feature-selection-pkls/{key}')

def AzBio_optimization():
    all_labels = [
        'hz125_R', 'hz125_L', 'hz250_R', 'hz250_L',
        'hz500_R', 'hz500_L', 'hz750_R', 'hz750_L',
        'hz1000_R', 'hz1000_L', 'hz1500_R', 'hz1500_L',
        'hz2000_R', 'hz2000_L', 'hz3000_R', 'hz3000_L',
        'hz4000_R', 'hz4000_L', 'hz6000_R', 'hz6000_L', 'hz8000_R', 'hz8000_L',
        'WRS_L', 'WRS_R', 'Age', 'AzBioQuiet_bi'
    ]

    nested_cross_optimize_bilateral(n_iters=100, k_outer=10, k_inner=10, all_labels=all_labels,smote=False,file_name='az',folder_name='AzBio-pkls-binary',target = 'AzBioQuiet_bi')
    Post_Iter_Processing(folder_name='AzBio-pkls-binary')





if __name__ == '__main__':
    # sixty_sixty_run()  #Run this to get 60/60 CM and information
    # Demographics_Table() #Run this to get demographics information
    # feature_set_optimize()
    # AzBio_optimization()
    key_set_optimization()
    # sixty_sixty_run_az_bio()





