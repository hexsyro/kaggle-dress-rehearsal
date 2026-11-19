
import time
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold


def runner(x,y):
    opt_start_time = time.time()
    def objective(trial):
        model = xgb.XGBClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 15),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.4, log=True),
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            random_state=42,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, x, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30)
    opt_time = time.time() - opt_start_time

    return opt_time, study

def save(fig, name):
    fig.savefig(f"outputs/figures/{name}.png", dpi=200, bbox_inches="tight")

def plot(study):
    ax = plot_optimization_history(study)
    ax.set_title("Optuna search progress")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("AUC (higher is better)")
    ax.axvline(9.5, color="gray", linestyle="--")
    ax.figure.tight_layout()
    save(ax.figure, "optuna_history")
    plt.show()


def main():
    print()
    print()
    x=pd.read_csv('data/processed/x.csv')
    y=pd.read_csv('data/processed/y.csv').squeeze()
    print("input and target files loaded successfully!")
    print()

    opt_time, study=runner(x,y)
    print("Best hyperparameters found: \n", study.best_params)
    print("Best AUC score: ", study.best_value)
    print(f"Optimization time: {opt_time:.2f} seconds")
    print()

    plot(study)    
    print('Done!')

if __name__ == "__main__":
    main()