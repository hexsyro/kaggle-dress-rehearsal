import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def split(df):
    x = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    return x, y


def encode(x):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = ohe.fit_transform(x[CATS])
    enc = pd.DataFrame(arr, columns=ohe.get_feature_names_out(CATS), index=x.index)
    return x.drop(columns=CATS).join(enc)


def scale(x_enc):
    scaler = StandardScaler()
    x_sc = x_enc.copy()
    x_sc[NUMS] = scaler.fit_transform(x_enc[NUMS])
    return x_sc


def main():
    df = pd.read_csv('data/raw/Telco_Customer_Churn.csv')
    print("Raw Dataset loaded successfully!")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    print("Fixed 'TotalCharges' type and null values.")

    x, y = split(df)
    assert y.notna().all(), "Churn mapping produced NaN, check the raw values"
    print("Created input and target data successfully!")

    x_enc = encode(x)
    print("Encoded categorical columns successfully!")

    x_sc = scale(x_enc)
    print("Scaled numeric columns successfully!")

    os.makedirs('data/processed', exist_ok=True)
    x.to_csv('data/processed/x_unenc.csv', index=False)
    x_enc.to_csv('data/processed/x_unsc.csv', index=False)
    y.to_csv('data/processed/y.csv', index=False)
    x_sc.to_csv('data/processed/x.csv', index=False)
    print("Saved 4 files into 'data/processed'. Done!")


if __name__ == "__main__":
    main()