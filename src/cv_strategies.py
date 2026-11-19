
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def kfold(x,y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results_kf = []
    for train_index, valid_index in kf.split(x):
        m=LogisticRegression(max_iter=1000)
        m.fit(x.iloc[train_index], y.iloc[train_index])
        proba=m.predict_proba(x.iloc[valid_index])[:,1]
        roc_auc = roc_auc_score(y.iloc[valid_index], proba)
        results_kf.append(roc_auc)
        print(f'Fold AUC: {roc_auc:.4f}')
    return results_kf, kf

def skfold(x,y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_skf = []
    for train_index, valid_index in skf.split(x,y):
        m2=LogisticRegression(max_iter=1000)
        m2.fit(x.iloc[train_index], y.iloc[train_index])
        proba=m2.predict_proba(x.iloc[valid_index])[:,1]
        roc_auc = roc_auc_score(y.iloc[valid_index], proba)
        results_skf.append(roc_auc)
        print(f'Fold AUC: {roc_auc:.4f}')
    return results_skf, skf

def props(x,y,kf,skf):
    kf_props = [y.iloc[val].mean() for _, val in kf.split(x)]
    skf_props = [y.iloc[val].mean() for _, val in skf.split(x, y)]

    kfold_std=np.std(kf_props)
    skfold_std=np.std(skf_props)
    ratio=kfold_std/skfold_std

    return ratio,kfold_std,skfold_std



def main():
    print()
    print()
    x=pd.read_csv('data/processed/x.csv')
    y=pd.read_csv('data/processed/y.csv').squeeze()
    print("input and target files loaded successfully!")
    print()

    results_kf, kf=kfold(x,y)
    print(f'Standard Deviation of AUC: {np.std(results_kf):.4f}')
    print(f'Mean AUC: {np.mean(results_kf):.4f}')
    print()

    results_skf, skf=skfold(x,y)
    print(f'Standard Deviation of AUC: {np.std(results_skf):.4f}')
    print(f'Mean AUC: {np.mean(results_skf):.4f}')
    print()

    print("how much the churn rate changes from fold to fold.")
    ratio,kfold_std,skfold_std=props(x,y,kf,skf)
    print("KFold std:", kfold_std)
    print("Stratified std:", skfold_std)
    print()
    print(f"Ratio is about {ratio}. Stratified folds are roughly {ratio} times more consistent, \n",
        f"which is where the “~{ratio}x tighter” in the posts came from.")
    print('Done!')

if __name__ == "__main__":
    main()

