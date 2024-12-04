import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# ================================================
# Bagian 1: Fungsi Preprocessing dan Pelatihan Model
# ================================================
def preprocess_and_train_model(dataset, target_column, model_filename, scaler_filename):
    # Menentukan kolom numerik
    numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns

    # Mengisi nilai yang hilang hanya pada kolom numerik
    dataset[numerical_columns] = dataset[numerical_columns].fillna(dataset[numerical_columns].mean())
    
    # Memisahkan fitur dan target
    x = dataset.drop(columns=[target_column, 'source'], axis=1)
    y = dataset[target_column]
    
    # Standarisasi data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=2)

    # Melatih model menggunakan SVM
    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)
    
    # Evaluasi model
    train_accuracy = accuracy_score(classifier.predict(x_train), y_train)
    test_accuracy = accuracy_score(classifier.predict(x_test), y_test)

    # Menampilkan hasil akurasi
    st.write(f'*Akurasi Data Training ({target_column})*: {train_accuracy:.2f}')
    st.write(f'*Akurasi Data Testing ({target_column})*: {test_accuracy:.2f}')

    # Simpan model dan scaler
    pickle.dump(classifier, open(model_filename, 'wb'))
    pickle.dump(scaler, open(scaler_filename, 'wb'))

# ================================================
# Bagian 2: Fungsi Prediksi
# ================================================
def predict_potability(input_data, model_filename, scaler_filename):
    # Load model dan scaler
    model = pickle.load(open(model_filename, 'rb'))
    scaler = pickle.load(open(scaler_filename, 'rb'))
    
    # Preprocessing data input
    input_array = np.array(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    
    # Prediksi
    prediction = model.predict(std_data)
    return prediction

# ================================================
# Bagian 3: Aplikasi Streamlit
# ================================================
st.title("Prediksi Kelayakan Air Minum dan Tanaman")

# Pilihan tab untuk pengguna
tab1, tab2 = st.tabs(["Training Model", "Prediksi Kelayakan"])

# ================================================
# Tab 1: Training Model
# ================================================
with tab1:
    st.header("Training Model")
    st.write("Unggah dataset Anda untuk melatih model.")

    # Upload dataset pertama
    dataset1_file = st.file_uploader("Upload dataset pertama (Kelayakan Air Minum)", type=["csv"])
    # Upload dataset kedua
    dataset2_file = st.file_uploader("Upload dataset kedua (Kelayakan Tanaman)", type=["csv"])

    if dataset1_file and dataset2_file:
        dataset1 = pd.read_csv(dataset1_file)
        dataset2 = pd.read_csv(dataset2_file)
        
        # Menambahkan kolom 'source' untuk membedakan dataset
        dataset1['source'] = 'minum'
        dataset2['source'] = 'tanaman'
        
        # Training model untuk dataset pertama
        st.subheader("Model untuk Kelayakan Air Minum")
        preprocess_and_train_model(dataset1, 'Potability', 'model_air_minum.sav', 'scaler_air_minum.sav')
        
        # Training model untuk dataset kedua
        st.subheader("Model untuk Kelayakan Tanaman")
        preprocess_and_train_model(dataset2, 'Potability', 'model_tanaman.sav', 'scaler_tanaman.sav')

# ================================================
# Tab 2: Prediksi Kelayakan
# ================================================
with tab2:
    st.header("Prediksi Kelayakan")
    st.write("Masukkan data air untuk memprediksi kelayakan.")

    # Input dari pengguna
    pH = st.number_input("pH", value=0.0)
    Hardness = st.number_input("Hardness", value=0.0)
    Solids = st.number_input("Solids", value=0.0)
    Chloramines = st.number_input("Chloramines", value=0.0)
    Sulfate = st.number_input("Sulfate", value=0.0)
    Conductivity = st.number_input("Conductivity", value=0.0)
    Organic_carbon = st.number_input("Organic Carbon", value=0.0)
    Trihalomethanes = st.number_input("Trihalomethanes", value=0.0)
    Turbidity = st.number_input("Turbidity", value=0.0)

    if st.button("Prediksi"):
        # Data input
        input_data = (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
        
        # Prediksi air minum
        prediksi_air = predict_potability(input_data, 'model_air_minum.sav', 'scaler_air_minum.sav')
        if prediksi_air[0] == 1:
            st.success("Air layak minum")
        else:
            st.error("Air tidak layak minum")
        
        # Prediksi air untuk tanaman
        prediksi_tanaman = predict_potability(input_data, 'model_tanaman.sav', 'scaler_tanaman.sav')
        if prediksi_tanaman[0] == 1:
            st.success("Air layak untuk tanaman")
        else:
            st.error("Air tidak layak untuk tanaman")