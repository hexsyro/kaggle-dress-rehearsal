
import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def grid_search(x,y):
    gs_results = []
    grid_start_time = time.time()
    m=xgb.XGBClassifier(random_state=42, eval_metric='auc')
    params={
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    GS=GridSearchCV(m, params, cv=StratifiedKFold(n_splits=5, shuffle=True, 
                                                random_state=42), scoring='roc_auc', n_jobs=-1)
    for train, test in GS.cv.split(x, y):
        GS.fit(x.iloc[train], y.iloc[train])
        GS_pred=GS.predict_proba(x.iloc[test])[:,1]
        GS_roc_auc=roc_auc_score(y.iloc[test], GS_pred)
        gs_results.append(round(GS_roc_auc, 4))
    grid_time = time.time() - grid_start_time
    return GS, gs_results, grid_time

def rand_search(x,y):
    random_start_time = time.time()
    rs_results = []
    m=xgb.XGBClassifier(random_state=42, eval_metric='auc')
    params={
        'n_estimators': range(100, 800, 10),  # RandomizedSearchCV can handle random distributions
        'max_depth': range(2, 15, 1),  # RandomizedSearchCV can handle ranges
        'learning_rate': np.random.uniform(0.005, 0.4, 100)  # RandomizedSearchCV can handle random distributions
    }
    RS=RandomizedSearchCV(m, params, n_iter=30, cv=StratifiedKFold(n_splits=5, shuffle=True, 
                                                        random_state=42), scoring='roc_auc', n_jobs=-1)
    for train, test in RS.cv.split(x, y):
        RS.fit(x.iloc[train], y.iloc[train])
        RS_pred=RS.predict_proba(x.iloc[test])[:,1]
        RS_roc_auc=roc_auc_score(y.iloc[test], RS_pred)
        rs_results.append(round(RS_roc_auc, 4))
    random_time = time.time() - random_start_time
    return RS, rs_results, random_time


def main():
    print()
    print()
    x=pd.read_csv('data/processed/x.csv')
    y=pd.read_csv('data/processed/y.csv').squeeze()
    print("input and target files loaded successfully!")
    print()

    GS, gs_results, grid_time=grid_search(x,y)
    print("GridSearchCV ran successfully!")
    print()

    RS, rs_results, random_time=rand_search(x,y)
    print("RandomizedSearchCV ran successfully!")
    print()

    print(f"Best parameters from Grid Search: \n{GS.best_params_}")
    print(f"Best parameters from Randomized Search: \n{RS.best_params_}")
    print()

    print(f"{GS.best_score_:.4f} is the best ROC AUC score from Grid Search")
    print(f"{RS.best_score_:.4f} is the best ROC AUC score from Randomized Search")
    print()

    print(f"Time taken for Grid Search: {grid_time:.2f} seconds")
    print(f"Time taken for Randomized Search: {random_time:.2f} seconds")
    print()

    print("ROC AUC scores Comparison Table:")
    tb=pd.DataFrame({
        'Grid Search': gs_results,
        'Randomized Search': rs_results
    })
    print(tb)

    print('Done!')

if __name__ == "__main__":
    main()

