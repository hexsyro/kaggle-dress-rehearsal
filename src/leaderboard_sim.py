import pandas as pd
import optuna
import xgboost as xgb
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
FIXED = dict(subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE)


def split(x, y, random_state):
    x_tmp, x_priv, y_tmp, y_priv = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state)
    x_train, x_pub, y_train, y_pub = train_test_split(
        x_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=random_state)
    print("rows -> train:", len(x_train), "public:", len(x_pub), "private:", len(x_priv))
    return x_train, y_train, x_pub, y_pub, x_priv, y_priv


def objective(trial, data):
    x_train, y_train, x_pub, y_pub = data
    model = xgb.XGBClassifier(
        max_depth=trial.suggest_int("max_depth", 3, 15),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.4, log=True),
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        **FIXED,
    )
    model.fit(x_train, y_train)
    return roc_auc_score(y_pub, model.predict_proba(x_pub)[:, 1])


def run_study(data):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(partial(objective, data=data), n_trials=50)

    for n in range(10, len(study.trials) + 1, 10):
        best = max(study.trials[:n], key=lambda t: t.value)
        print(n, round(best.value, 4), best.params)
    return study


def main():
    x = pd.read_csv("data/processed/x.csv")
    y = pd.read_csv("data/processed/y.csv").squeeze()
    print("input and target files loaded successfully!")

    x_train, y_train, x_pub, y_pub, x_priv, y_priv = split(x, y, RANDOM_STATE)

    study = run_study((x_train, y_train, x_pub, y_pub))
    print("Best hyperparameters found:\n", study.best_params)

    best_model = xgb.XGBClassifier(**study.best_params, **FIXED)
    best_model.fit(x_train, y_train)

    pub_auc = roc_auc_score(y_pub, best_model.predict_proba(x_pub)[:, 1])
    priv_auc = roc_auc_score(y_priv, best_model.predict_proba(x_priv)[:, 1])
    print(f"public AUC  (tuned against): {pub_auc:.4f}")
    print(f"private AUC (never used):    {priv_auc:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()