import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def run_preprocessing():
    # Mendefinisikan path file
    # Menunjuk ke credit_risk_dataset.csv di folder utama
    raw_data_path = os.path.join(os.path.dirname(__file__), '../credit_risk_dataset.csv')
    # Menunjuk ke credit_risk_clean.csv di dalam folder preprocessing
    processed_data_path = os.path.join(os.path.dirname(__file__), 'credit_risk_clean.csv')

    print("Memuat data raw...")
    # Menambahkan penanganan eror jika script dijalankan dengan path lokal yang berbeda
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        df = pd.read_csv('credit_risk_dataset.csv')

    # Menangani Outlier
    df_clean = df[(df['person_age'] <= 80) & (df['person_income'] <= 4000000)].copy()

    # Memisahkan Fitur (X) dan Target (y)
    X = df_clean.drop('loan_status', axis=1)
    y = df_clean['loan_status'].reset_index(drop=True)

    # Setup Pipeline Scikit-Learn
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Pipeline untuk data numerik (mengisi nilai kosong dengan median + standarisasi)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk data kategorikal (mengisi nilai kosong dengan modus + One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Menggabungkan kedua pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Eksekusi Transformasi
    print("Menjalankan preprocessing pipeline...")
    X_transformed = preprocessor.fit_transform(X)
    
    # Mengambil nama kolom baru setelah proses One-Hot Encoding
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)

    # Menggabungkan kembali menjadi DataFrame utuh
    df_ready = pd.DataFrame(X_transformed, columns=all_feature_names)
    df_ready['loan_status'] = y

    # Menyimpan hasil otomatisasi menjadi file CSV baru
    # Jika dijalankan melalui GitHub Actions, path ini akan memastikan file tersimpan di tempat yang benar
    try:
        df_ready.to_csv(processed_data_path, index=False)
        print(f"Preprocessing selesai! Data siap latih disimpan di: {processed_data_path}")
    except OSError:
        df_ready.to_csv('credit_risk_clean.csv', index=False)
        print("Preprocessing selesai! Data siap latih disimpan sebagai credit_risk_clean.csv")

if __name__ == "__main__":
    run_preprocessing()