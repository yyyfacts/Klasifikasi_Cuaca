import streamlit as st
import pandas as pd
import numpy as np
import json # Untuk parsing JSON string
import time # Untuk mengukur waktu pelatihan

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer, precision_score, recall_score
from imblearn.over_sampling import SMOTE # Untuk penanganan ketidakseimbangan kelas

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier # KNN Model
from xgboost import XGBClassifier

# --- Fungsi Pra-pemrosesan Data dan Pelatihan Model ---
@st.cache_resource # Cache the model and scaler to avoid re-training on every rerun
def train_model():
    with st.spinner('Memuat dan melatih model cuaca... Ini mungkin memakan waktu beberapa saat.'):
        # === Load Dataset ===
        try:
            # Asumsi file CSV ada di direktori yang sama dengan app.py
            df = pd.read_csv('weather data classification.csv')
            st.success("✅ Dataset berhasil dimuat.")
        except FileNotFoundError:
            st.error("❌ Error: File 'weather data classification.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan 'app.py'.")
            st.stop() # Hentikan aplikasi jika file tidak ditemukan

        # === Data Preprocessing ===
        # Membersihkan Kolom 'Wind Speed'
        df['Wind Speed'] = df['Wind Speed'].astype(str).str.lower()
        def convert_to_kmh(value):
            if 'km/s' in value:
                return float(value.replace('km/s', '').strip()) * 3600
            elif 'm/h' in value:
                return float(value.replace('m/h', '').strip()) / 1000
            else:
                try:
                    return float(value)
                except ValueError:
                    return np.nan
        df['Wind Speed'] = df['Wind Speed'].apply(convert_to_kmh)

        # Membersihkan Kolom 'Precipitation (%)'
        df['Precipitation (%)'] = df['Precipitation (%)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        df.loc[df['Precipitation (%)'] > 100, 'Precipitation (%)'] = np.nan

        # Membersihkan Kolom 'Cloud Cover'
        df['Cloud Cover'] = df['Cloud Cover'].astype(str).str.lower()
        df['Cloud Cover'] = df['Cloud Cover'].replace({
            'partly cloddy': 'partly cloudy',
            'party cloudy': 'partly cloudy',
            'clr': 'clear',
            'cleer': 'clear',
            'overcast': 'over cast',
            'ovrcst': 'over cast',
            'overcst': 'over cast'
        })

        # Menangani Outlier 'Atmospheric Pressure'
        df.loc[(df['Atmospheric Pressure'] > 1050) | (df['Atmospheric Pressure'] < 950), 'Atmospheric Pressure'] = np.nan

        # Membersihkan Kolom 'Season'
        df['Season'] = df['Season'].astype(str).str.lower()
        df['Season'] = df['Season'].replace({
            'wenter': 'winter',
            'wentar': 'winter',
            'wintar': 'winter',
            'it rains every day....': 'winter',
            'it is so hot outside': 'summer',
            'i hate summer': 'summer',
            'sumer': 'summer',
            'autum': 'autumn',
            'trees lose their leaves in this time': 'autumn',
            'fall': 'autumn',
            'you can call it autumn or fall ': 'autumn'
        })

        # Ekstraksi Temperature dan Humidity
        def extract_temp_humidity(json_str):
            if pd.isna(json_str) or not isinstance(json_str, str):
                return np.nan, np.nan
            try:
                data = json.loads(json_str)
                temp = data.get("Temperature", {}).get("value")
                humidity = data.get("Humidity", {}).get("value")
                return temp, humidity
            except json.JSONDecodeError:
                try:
                    temp_str = json_str.split(',')[0].split(':')[-1].strip()
                    humidity_str = json_str.split(',')[-2].split(':')[-1].strip()
                    return float(temp_str), float(humidity_str)
                except (IndexError, ValueError):
                    return np.nan, np.nan
            except AttributeError:
                return np.nan, np.nan

        df[['Temperature', 'Humidity']] = df['{Temperature},{Humdity}'].apply(
            lambda x: pd.Series(extract_temp_humidity(x))
        )

        # Menangani outlier Temperature dan Humidity
        df.loc[df['Temperature'] > 65, 'Temperature'] = np.nan
        df.loc[df['Humidity'] > 100, 'Humidity'] = np.nan
        df.drop(columns=['{Temperature},{Humdity}'], inplace=True)

        # Penanganan Nilai Hilang
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df.loc[:, col] = df[col].fillna(mean_val)
        for col in df.select_dtypes(include='object').columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df.loc[:, col] = df[col].fillna(mode_val)

        # === Encoding & Pemisahan Fitur-Target ===
        # Encoding Label Target
        le = LabelEncoder()
        df['weather_encoded'] = le.fit_transform(df['Weather Type'])
        label_classes = le.classes_ # Simpan kelas label

        # One-Hot Encoding untuk Fitur Kategorikal
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if 'Weather Type' in categorical_cols:
            categorical_cols.remove('Weather Type')
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Pisahkan Fitur (X) dan Target (y)
        X = df_encoded.drop(columns=['Weather Type', 'weather_encoded'], errors='ignore')
        y = df_encoded['weather_encoded']
        feature_columns = X.columns.tolist() # Simpan daftar kolom fitur

        # === Split Data & Scaling ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Penanganan Ketidakseimbangan Data dengan SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Normalisasi Data Numerik
        numeric_cols = X_train_resampled.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        X_train_scaled_numeric = scaler.fit_transform(X_train_resampled[numeric_cols])
        X_train_final = pd.DataFrame(X_train_scaled_numeric, columns=numeric_cols, index=X_train_resampled.index)

        categorical_encoded_cols = [col for col in X.columns if col not in numeric_cols]
        X_train_final = pd.concat([X_train_final, X_train_resampled[categorical_encoded_cols]], axis=1)
        X_train_final = X_train_final[X.columns] # Pastikan urutan kolom sama

        # === Pembangunan dan Pelatihan Model XGBoost (Tuned) ===
        # Gunakan parameter terbaik dari notebook Anda
        # (contoh parameter terbaik dari analisis sebelumnya)
        best_params_xgb = {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}
        model = XGBClassifier(
            **best_params_xgb,
            objective='multi:softmax',
            num_class=len(y.unique()),
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist'
        )
        model.fit(X_train_final, y_train_resampled)
        st.success("✅ Pelatihan model selesai!")

    return model, scaler, feature_columns, label_classes

# Latih model (atau muat dari cache jika sudah dijalankan)
model, scaler, feature_columns, label_classes = train_model()

# Judul Aplikasi
st.title('Aplikasi Prediksi Jenis Cuaca')
st.write('Masukkan parameter cuaca untuk memprediksi jenis cuaca (Cloudy, Rainy, Snowy, Sunny).')

# --- Fungsi Pra-pemrosesan untuk Input Pengguna ---
def preprocess_input(input_data, feature_columns, scaler):
    df_input = pd.DataFrame([input_data])

    # Konversi 'Cloud Cover', 'Season', 'Location' ke lowercase untuk konsistensi
    df_input['Cloud Cover'] = df_input['Cloud Cover'].str.lower()
    df_input['Season'] = df_input['Season'].str.lower()
    df_input['Location'] = df_input['Location'].str.lower()

    # Identifikasi kolom numerik dan kategorikal dari feature_columns
    numeric_cols_base = [col for col in feature_columns if '_' not in col and col not in ['Cloud Cover', 'Season', 'Location']]

    # Lakukan One-Hot Encoding untuk kolom kategorikal
    df_encoded = pd.get_dummies(df_input, columns=['Cloud Cover', 'Season', 'Location'], drop_first=True)

    # Buat DataFrame akhir dengan semua kolom yang diharapkan model, diinisialisasi dengan 0
    final_input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Isi nilai numerik
    for col in numeric_cols_base:
        if col in df_encoded.columns:
            final_input_df[col] = df_encoded[col]

    # Isi nilai one-hot encoded
    for col in df_encoded.columns:
        if col in feature_columns and col not in numeric_cols_base:
             final_input_df[col] = df_encoded[col]

    # Scaling kolom numerik
    numeric_cols_for_scaling = [col for col in numeric_cols_base if col in final_input_df.columns]
    final_input_df[numeric_cols_for_scaling] = scaler.transform(final_input_df[numeric_cols_for_scaling])

    return final_input_df

# --- Input Pengguna ---
st.header('Parameter Cuaca')

# Input numerik
wind_speed = st.number_input('Kecepatan Angin (KM/H)', min_value=0.0, max_value=200.0, value=10.0, step=0.1)
precipitation = st.number_input('Curah Hujan (%)', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
atmospheric_pressure = st.number_input('Tekanan Atmosfer (hPa)', min_value=950.0, max_value=1050.0, value=1010.0, step=0.1) # Adjusted min/max based on preprocessing
uv_index = st.number_input('Indeks UV', min_value=0, max_value=15, value=5, step=1)
visibility = st.number_input('Jarak Pandang (KM)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
temperature = st.number_input('Suhu (°C)', min_value=-20.0, max_value=65.0, value=25.0, step=0.1) # Adjusted max based on preprocessing
humidity = st.number_input('Kelembaban (%)', min_value=0.0, max_value=100.0, value=70.0, step=0.1)

# Input kategorikal
cloud_cover = st.selectbox('Tutupan Awan', ('partly cloudy', 'clear', 'overcast'))
season = st.selectbox('Musim', ('winter', 'spring', 'summer', 'autumn'))
location = st.selectbox('Lokasi', ('inland', 'mountain', 'coastal'))

# Tombol Prediksi
if st.button('Prediksi Cuaca'):
    # Kumpulkan input dalam bentuk dictionary
    input_data = {
        'Wind Speed': wind_speed,
        'Precipitation (%)': precipitation,
        'Atmospheric Pressure': atmospheric_pressure,
        'UV Index': uv_index,
        'Visibility (km)': visibility,
        'Temperature': temperature,
        'Humidity': humidity,
        'Cloud Cover': cloud_cover,
        'Season': season,
        'Location': location
    }

    # Pra-pemrosesan input
    processed_input = preprocess_input(input_data, feature_columns, scaler)

    # Melakukan prediksi
    prediction_encoded = model.predict(processed_input)
    prediction_label = label_classes[prediction_encoded[0]]

    # Menampilkan hasil
    st.subheader('Hasil Prediksi:')
    st.success(f'Jenis Cuaca yang Diprediksi: **{prediction_label}**')

    st.write("---")
    st.write("Catatan: Prediksi ini didasarkan pada model Machine Learning yang dilatih dengan dataset cuaca yang telah disediakan.")