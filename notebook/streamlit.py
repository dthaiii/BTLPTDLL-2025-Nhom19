import streamlit as st
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel

@st.cache_resource
def get_spark_and_model():
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("AQI_App") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.port", "0") \
        .config("spark.blockManager.port", "0") \
        .config("spark.ui.port", "0") \
        .config("spark.network.timeout", "1200s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()
    
    model = PipelineModel.load("./linear_regression_aqi")
    return spark, model

spark, model = get_spark_and_model()

def get_aqi_status(aqi_value):
    if aqi_value <= 50:
        return "Tốt"
    elif aqi_value <= 100:
        return "Trung bình"
    elif aqi_value <= 150:
        return "Kém"
    elif aqi_value <= 200:
        return "Xấu"
    elif aqi_value <= 300:
        return "Rất xấu"
    else: 
        return "Nguy hại"

st.title("Dự đoán chỉ số AQI (Mô hình Hồi quy tuyến tính)")

col1, col2 = st. columns(2)
with col1:
    Wind = st.number_input("Wind (m/s)", 0.0, 50.0, 2.5)
    CO = st.number_input("CO (µg/m³)", 0.0, 10000.0, 500.0)
    Dew = st.number_input("Dew (°C)", -20.0, 40.0, 15.0)
    Humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    NO2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 40.0)
    O3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 50.0)

with col2:
    PM10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 50.0)
    PM25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 25.0)
    Pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    SO2 = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 10.0)
    Temperature = st.number_input("Temperature (°C)", -20.0, 50.0, 25.0)

if st.button("Dự đoán AQI"):
    data_dict = {
        "Wind": float(Wind),
        "CO": float(CO),
        "Dew": float(Dew),
        "Humidity": float(Humidity),
        "NO2": float(NO2),
        "O3": float(O3),
        "PM10": float(PM10),
        "PM25": float(PM25),
        "Pressure": float(Pressure),
        "SO2": float(SO2),
        "Temperature": float(Temperature)
    }
    
    try:
        input_data = spark.createDataFrame([data_dict])
        predictions = model.transform(input_data).collect()
        
        if predictions: 
            predicted_aqi = float(predictions[0]['prediction'])
            aqi_status = get_aqi_status(predicted_aqi)
            st.success(f"Dự đoán chỉ số AQI: {predicted_aqi:.2f} ({aqi_status})")
        else:
            st.error("Không nhận được kết quả dự đoán.")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")