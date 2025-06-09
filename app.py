import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json # Import library json untuk menguraikan string JSON

# --- Konfigurasi Halaman Streamlit (HARUS JADI PERINTAH STREAMLIT PERTAMA) ---
st.set_page_config(layout="wide")

# --- Fungsi Konversi Kecepatan Angin (dari notebook) ---
def convert_to_kmh(value):
    value_lower = str(value).lower()
    if 'km/s' in value_lower:
        return float(value_lower.replace('km/s', '').strip()) * 3600
    elif 'm/h' in value_lower:
        return float(value_lower.replace('m/h', '').strip()) / 1000
    else:
        try:
            return float(value_lower)
        except ValueError:
            return np.nan # Mengembalikan NaN jika tidak bisa dikonversi

# --- 1. Data Preprocessing dan Pelatihan Model (sesuai Klasifikasi_Cuaca (1).ipynb) ---
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("weather data classification.csv")
    except FileNotFoundError:
        st.error("Error: File 'weather data classification.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        st.stop()

    # --- Pembersihan dan Transformasi Data (sesuai notebook) ---
    # Kolom 'Wind Speed'
    df['Wind Speed'] = df['Wind Speed'].apply(convert_to_kmh)

    # Kolom 'Precipitation (%)'
    df['Precipitation (%)'] = df['Precipitation (%)'].astype(str).str.extract('(\\d+\\.\\d+)?')
    df['Precipitation (%)'] = pd.to_numeric(df['Precipitation (%)'])
    df.loc[df['Precipitation (%)'] > 100, 'Precipitation (%)'] = np.nan # Menangani outlier > 100%

    # Kolom 'Cloud Cover'
    df['Cloud Cover'] = df['Cloud Cover'].astype(str).str.lower()
    df['Cloud Cover'] = df['Cloud Cover'].replace({
        'partly cloddy': 'partly cloudy',
        'party cloudy': 'partly cloudy',
        'clr': 'clear',
        'cleer': 'clear',
        'overcst': 'over cast',
        'ovrcst': 'over cast',
        'overcast': 'over cast',
    })

    # Kolom 'Atmospheric Pressure'
    df.loc[(df['Atmospheric Pressure'] > 1050) | (df['Atmospheric Pressure'] < 950), 'Atmospheric Pressure'] = np.nan

    # Kolom 'Season'
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

    # Kolom '{Temperature},{Humdity}'
    # Ekstraksi Temperature dan Humidity dari kolom JSON string
    # Mengganti single quote dengan double quote agar sesuai format JSON yang valid
    def parse_temp_humidity(s):
        try:
            # Replace single quotes for valid JSON parsing
            json_str = s.replace("'", "\"")
            data = json.loads(json_str)
            temp = data['Temperature']['value']
            humidity = data['Humidity']['value']
            return temp, humidity
        except (json.JSONDecodeError, KeyError, AttributeError):
            return np.nan, np.nan

    df[['Temperature', 'Humidity']] = df['{Temperature},{Humdity}'].apply(lambda x: pd.Series(parse_temp_humidity(x)))
    
    df['Temperature'] = pd.to_numeric(df['Temperature'])
    df.loc[df['Temperature'] > 65, 'Temperature'] = np.nan

    df['Humidity'] = pd.to_numeric(df['Humidity'])
    df.loc[df['Humidity'] > 100, 'Humidity'] = np.nan

    # Menghapus kolom asli setelah ekstraksi
    df.drop(columns=['{Temperature},{Humdity}'], inplace=True)

    # Penanganan nilai hilang (setelah semua pembersihan kolom spesifik)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    # Identifikasi dan Tangani Kolom Kategorikal (Encoding)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    target_column_name = 'Weather Type'
    le = LabelEncoder()
    df['weather_encoded'] = le.fit_transform(df[target_column_name])
    
    if target_column_name in categorical_cols:
        categorical_cols.remove(target_column_name)

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Pemisahan Fitur (X) dan Target (y)
    X = df_encoded.drop(columns=[target_column_name, 'weather_encoded'], axis=1, errors='ignore')
    y = df_encoded['weather_encoded']

    # Pemisahan Data Latih dan Uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Penskalaan Fitur Numerik
    numeric_cols_after_encoding = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols_after_encoding])
    X_test_scaled = scaler.transform(X_test[numeric_cols_after_encoding])

    X_train_final = pd.DataFrame(X_train_scaled, columns=numeric_cols_after_encoding, index=X_train.index)
    X_test_final = pd.DataFrame(X_test_scaled, columns=numeric_cols_after_encoding, index=X_test.index)
    
    # Periksa apakah ada kolom non-numerik yang tersisa (one-hot encoded boolean)
    # Ini penting karena get_dummies bisa menghasilkan kolom boolean (bukan float/int)
    non_numeric_cols_train = X_train.select_dtypes(include='bool').columns
    non_numeric_cols_test = X_test.select_dtypes(include='bool').columns

    if not non_numeric_cols_train.empty:
        X_train_final = pd.concat([X_train_final, X_train[non_numeric_cols_train]], axis=1)
        X_test_final = pd.concat([X_test_final, X_test[non_numeric_cols_test]], axis=1)
    
    # Pelatihan Model (Menggunakan parameter terbaik dari notebook)
    # Random Forest (Tuned)
    rf_best = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, n_estimators=100, random_state=42)
    rf_best.fit(X_train_final, y_train)

    # XGBoost (Tuned)
    xgb_best = XGBClassifier(colsample_bytree=1.0, learning_rate=0.1, max_depth=3, n_estimators=100,
                             objective='multi:softmax', num_class=len(y.unique()),
                             use_label_encoder=False, eval_metric='mlogloss', tree_method='hist', random_state=42)
    xgb_best.fit(X_train_final, y_train)

    # Naive Bayes (Tuned)
    gnb_best = GaussianNB(var_smoothing=1.0)
    gnb_best.fit(X_train_final, y_train)

    tuned_models = {
        "Random Forest (Tuned)": rf_best,
        "XGBoost (Tuned)": xgb_best,
        "Naive Bayes (Tuned)": gnb_best
    }

    # Mengembalikan semua kolom yang digunakan dalam model setelah one-hot encoding dan penambahan fitur baru
    all_model_features_after_encoding = X_train_final.columns.tolist()

    return df, X, y, X_train_final, X_test_final, y_train, y_test, scaler, le, tuned_models, all_model_features_after_encoding

# Memuat dan pra-pemrosesan data serta melatih model saat aplikasi dimulai
df_original, X_all, y_all, X_train_final, X_test_final, y_train, y_test, scaler, le, tuned_models, all_model_features_after_encoding = load_and_preprocess_data()


# --- Aplikasi Streamlit ---

st.title("Aplikasi Klasifikasi Cuaca")

menu = ["Tentang Aplikasi dan Tujuan", "Prediksi Jenis Cuaca", "Evaluasi Model"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

# --- 1. Tentang Aplikasi dan Tujuan ---
if choice == "Tentang Aplikasi dan Tujuan":
    st.header("Tentang Aplikasi Klasifikasi Cuaca")
    st.write("""
        Aplikasi ini dibangun untuk mengklasifikasikan jenis cuaca berdasarkan beberapa parameter lingkungan.
        Dengan memanfaatkan berbagai algoritma Machine Learning, kami berupaya memprediksi apakah cuaca di suatu lokasi
        termasuk kategori 'Cloudy', 'Rainy', 'Snowy', atau 'Sunny'.
    """)
    st.subheader("Tujuan Aplikasi")
    st.write("""
        * **Memahami Data Cuaca:** Melakukan eksplorasi dan pra-pemrosesan data cuaca untuk memahami karakteristiknya.
        * **Membangun Model Prediktif:** Mengembangkan model Machine Learning yang dapat memprediksi jenis cuaca
          berdasarkan fitur-fitur yang tersedia.
        * **Membandingkan Performa Model:** Mengevaluasi dan membandingkan performa berbagai model (Random Forest, XGBoost, Naive Bayes)
          untuk menemukan model terbaik.
        * **Antarmuka Interaktif:** Menyediakan antarmuka yang mudah digunakan bagi pengguna untuk memasukkan parameter
          dan mendapatkan prediksi jenis cuaca secara instan.
    """)
    st.subheader("Bagaimana Aplikasi Bekerja?")
    st.write("""
        1. **Pra-pemrosesan Data:** Data cuaca mentah dibersihkan, nilai-nilai yang hilang ditangani, dan fitur kategorikal
           diubah menjadi format numerik (one-hot encoding) yang dapat dipahami oleh model Machine Learning.
           Fitur numerik juga diskalakan agar memiliki rentang yang seragam.
        2. **Pelatihan Model:** Beberapa model Machine Learning dilatih menggunakan data yang telah diproses.
           Model-model ini belajar pola dari data historis untuk membuat prediksi.
        3. **Prediksi:** Ketika pengguna memasukkan parameter cuaca baru, aplikasi akan memproses input tersebut
           dengan cara yang sama seperti data pelatihan, kemudian menggunakan model yang telah dilatih untuk
           memprediksi jenis cuaca yang paling mungkin.
        4. **Evaluasi:** Bagian evaluasi model menampilkan metrik kinerja dari model yang telah dilatih,
           membantu memahami seberapa baik model tersebut bekerja.
    """)

# --- 2. Prediksi Jenis Cuaca ---
elif choice == "Prediksi Jenis Cuaca":
    st.header("Prediksi Jenis Cuaca")
    st.write("Silakan masukkan parameter cuaca di bawah ini untuk memprediksi jenis cuaca.")

    # Input dari pengguna
    col1, col2 = st.columns(2)

    with col1:
        # Input Kecepatan Angin
        st.write("#### Kecepatan Angin (KM/S)")
        wind_speed_value_raw = st.text_input("Masukkan Kecepatan Angin (misal: 8.5 atau 0.002638888888888889KM/S):")
        
        # Input Presipitasi
        st.write("#### Presipitasi (%)")
        precipitation_value_raw = st.text_input("Masukkan Presipitasi (%) (misal: 71.0):")
        
        # Input Tutupan Awan
        cloud_cover_unique_values = df_original['Cloud Cover'].unique().tolist()
        cloud_cover = st.selectbox("Tutupan Awan:", cloud_cover_unique_values)

    with col2:
        # Input Tekanan Atmosfer
        atmospheric_pressure_min = float(df_original['Atmospheric Pressure'].min())
        atmospheric_pressure_max = float(df_original['Atmospheric Pressure'].max())
        atmospheric_pressure_mean = float(df_original['Atmospheric Pressure'].mean())
        atmospheric_pressure = st.number_input("Tekanan Atmosfer:", min_value=atmospheric_pressure_min, max_value=atmospheric_pressure_max, value=atmospheric_pressure_mean, step=0.1)
        
        # Input Indeks UV
        uv_index_min = int(df_original['UV Index'].min())
        uv_index_max = int(df_original['UV Index'].max())
        uv_index_mean = int(df_original['UV Index'].mean())
        uv_index = st.slider("Indeks UV:", min_value=uv_index_min, max_value=uv_index_max, value=uv_index_mean)
        
        # Input Musim
        season_unique_values = df_original['Season'].unique().tolist()
        season = st.selectbox("Musim:", season_unique_values)
        
        # Input Jarak Pandang
        visibility_min = float(df_original['Visibility (km)'].min())
        visibility_max = float(df_original['Visibility (km)'].max())
        visibility_mean = float(df_original['Visibility (km)'].mean())
        visibility = st.number_input("Jarak Pandang (km):", min_value=visibility_min, max_value=visibility_max, value=visibility_mean, step=0.1)
        
        # Input Lokasi
        location_unique_values = df_original['Location'].unique().tolist()
        location = st.selectbox("Lokasi:", location_unique_values)
        
        # Input Suhu (dari kolom Temperature yang sudah diekstrak)
        temperature_min = float(df_original['Temperature'].min())
        temperature_max = float(df_original['Temperature'].max())
        temperature_mean = float(df_original['Temperature'].mean())
        temperature_c = st.number_input("Suhu (Celcius):", min_value=temperature_min, max_value=temperature_max, value=temperature_mean, step=0.1)
        
        # Input Kelembaban (dari kolom Humidity yang sudah diekstrak)
        humidity_min = int(df_original['Humidity'].min())
        humidity_max = int(df_original['Humidity'].max())
        humidity_mean = int(df_original['Humidity'].mean())
        humidity_pct = st.slider("Kelembaban (%):", min_value=humidity_min, max_value=humidity_max, value=humidity_mean)

    # Tombol Prediksi
    if st.button("Prediksi Cuaca"):
        # --- Pra-pemrosesan input pengguna agar sesuai dengan data pelatihan ---
        # 1. Konversi Kecepatan Angin dan Presipitasi
        wind_speed_processed = convert_to_kmh(wind_speed_value_raw)
        
        try:
            precipitation_processed = float(precipitation_value_raw)
            if precipitation_processed > 100 or precipitation_processed < 0:
                st.error("Presipitasi harus dalam rentang 0-100%.")
                st.stop()
        except ValueError:
            st.error("Presipitasi harus berupa angka.")
            st.stop()
        
        # 2. Tangani Outlier untuk Suhu (sesuai notebook)
        if temperature_c > 65:
            st.warning("Suhu melebihi batas data training (65°C) dan akan diinterpretasikan sebagai outlier.")
            temperature_c = df_original['Temperature'].mean() # Menggunakan mean dari data training

        # 3. Tangani Outlier untuk Kelembaban (sesuai notebook)
        if humidity_pct > 100:
            st.warning("Kelembaban melebihi batas data training (100%) dan akan diinterpretasikan sebagai outlier.")
            humidity_pct = df_original['Humidity'].mean() # Menggunakan mean dari data training

        # Buat DataFrame input dari pengguna
        input_data = {
            'Wind Speed': [wind_speed_processed],
            'Precipitation (%)': [precipitation_processed],
            'Cloud Cover': [cloud_cover],
            'Atmospheric Pressure': [atmospheric_pressure],
            'UV Index': [uv_index],
            'Season': [season],
            'Visibility (km)': [visibility],
            'Location': [location],
            'Temperature': [temperature_c],
            'Humidity': [humidity_pct]
        }
        input_df = pd.DataFrame(input_data)

        # 4. One-Hot Encode Fitur Kategorikal dari input_df
        #    Pastikan kolom-kolom one-hot encoded pada input_df sesuai dengan all_model_features_after_encoding
        
        # Identifikasi kolom kategorikal yang akan di-OHE dari input_df (sesuai load_and_preprocess_data)
        # Yaitu: 'Cloud Cover', 'Season', 'Location'
        categorical_features_for_ohe_from_input = ['Cloud Cover', 'Season', 'Location']
        
        # One-hot encode input_df, membuat kolom dummy untuk semua kategori yang *mungkin* ada di data asli
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_features_for_ohe_from_input, drop_first=True)

        # 5. Gabungkan fitur numerik yang diskalakan dengan fitur one-hot encoded
        # Buat DataFrame kosong dengan semua kolom yang diharapkan oleh model (X_train_final.columns)
        processed_input = pd.DataFrame(0, index=input_df.index, columns=all_model_features_after_encoding)
        
        # Kolom numerik yang akan diskalakan (sesuai load_and_preprocess_data)
        numeric_features_for_scaling = [
            'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 
            'Visibility (km)', 'Temperature', 'Humidity'
        ]

        # Penskalaan fitur numerik dari input_df
        # Penting: Hanya gunakan scaler.transform, jangan fit ulang!
        input_df_numeric_scaled = scaler.transform(input_df[numeric_features_for_scaling])
        input_df_numeric_scaled_df = pd.DataFrame(input_df_numeric_scaled, columns=numeric_features_for_scaling, index=input_df.index)

        # Isi kolom numerik yang diskalakan ke processed_input
        for col in numeric_features_for_scaling:
            if col in processed_input.columns:
                processed_input[col] = input_df_numeric_scaled_df[col]

        # Isi kolom one-hot encoded ke processed_input
        for col in input_df_encoded.columns:
            if col in processed_input.columns:
                processed_input[col] = input_df_encoded[col]
            # else:
            #     st.warning(f"Kolom one-hot encoded '{col}' dari input tidak ditemukan di fitur model.")

        # Prediksi dengan model terbaik (misal: Random Forest)
        model_to_use = tuned_models["Random Forest (Tuned)"]
        prediction_encoded = model_to_use.predict(processed_input)
        
        # Dekode hasil prediksi
        predicted_weather = le.inverse_transform(prediction_encoded)

        st.subheader("Hasil Prediksi:")
        st.success(f"Jenis Cuaca yang Diprediksi: **{predicted_weather[0]}**")
        st.write("Catatan: Akurasi prediksi bergantung pada kualitas data input dan performa model.")


# --- 3. Evaluasi Model ---
elif choice == "Evaluasi Model":
    st.header("Evaluasi Model")
    st.write("Berikut adalah metrik evaluasi dan matriks kebingungan untuk model-model yang telah disetel.")

    metrics_summary = []

    for name, model in tuned_models.items():
        st.subheader(f"Model: {name}")
        y_pred = model.predict(X_test_final)

        acc = accuracy_score(y_test, y_pred)
        # Menggunakan zero_division=0 untuk menghindari warning jika ada kelas tanpa prediksi/true samples
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) 
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.write(f"**Akurasi:** {acc:.4f}")
        st.write(f"**Presisi (Weighted):** {prec:.4f}")
        st.write(f"**Recall (Weighted):** {rec:.4f}")
        st.write(f"**F1-Score (Weighted):** {f1:.4f}")

        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, target_names=le.classes_))

        # Confusion Matrix
        st.write("Matriks Kebingungan:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title(f'Matriks Kebingungan - {name}')
        ax.set_xlabel('Diprediksi')
        ax.set_ylabel('Aktual')
        st.pyplot(fig)
        plt.close(fig) # Penting untuk menutup plot agar tidak tumpang tindih

        metrics_summary.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

    st.subheader("Ringkasan Performa Model")
    df_metrics = pd.DataFrame(metrics_summary)
    st.dataframe(df_metrics.set_index('Model').round(4))

    st.write("Visualisasi Perbandingan Performa Model:")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    df_metrics.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax_bar)
    ax_bar.set_title('Perbandingan Performa Model')
    ax_bar.set_ylabel('Skor')
    ax_bar.set_ylim(0, 1)
    ax_bar.grid(axis='y', linestyle='--')
    ax_bar.tick_params(axis='x', rotation=15)
    st.pyplot(fig_bar)
    plt.close(fig_bar)