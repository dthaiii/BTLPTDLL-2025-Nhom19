import streamlit as st
import os
import sys
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel

# 1. Khai báo Python Path (Bắt buộc trên Windows)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 2. Sử dụng Cache để chỉ khởi tạo Spark và Load Model 1 lần duy nhất
@st.cache_resource
@st.cache_resource
def get_spark_and_model():
    # 1. Khai báo Python Path chính xác
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # 2. Khởi tạo Spark với cấu hình "An toàn tối đa" cho Windows
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("AQI_App") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.network.timeout", "1000s") \
        .getOrCreate()
    
    # Load model
    model = PipelineModel.load("./linear_regression_aqi")
    return spark, model

# Khởi tạo
spark, model = get_spark_and_model()

# Hàm đánh giá chất lượng không khí
def get_aqi_status(aqi_value):
    if aqi_value <= 50: return "Tốt"
    elif aqi_value <= 100: return "Trung bình"
    elif aqi_value <= 150: return "Kém"
    elif aqi_value <= 200: return "Xấu"
    elif aqi_value <= 300: return "Rất xấu"
    else: return "Nguy hại"

st.title("Dự đoán chỉ số AQI (Mô hình Hồi quy tuyến tính)")

# Nhập liệu
col1, col2 = st.columns(2)
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
    # Tạo dictionary dữ liệu với tên cột khớp CHÍNH XÁC với Model (PM2.5 có dấu chấm)
    data_dict = {
        "Wind": float(Wind),
        "CO": float(CO),
        "Dew": float(Dew),
        "Humidity": float(Humidity),
        "NO2": float(NO2),
        "O3": float(O3),
        "PM10": float(PM10),
        "PM25": float(PM25),  # Chỗ này rất quan trọng: PM2.5 thay vì PM25
        "Pressure": float(Pressure),
        "SO2": float(SO2),
        "Temperature": float(Temperature)
    }
    
    # Tạo DataFrame từ 1 hàng dữ liệu
    input_data = spark.createDataFrame([Row(**data_dict)])

    try:
        predictions = model.transform(input_data)
        # Sử dụng .first() thay cho .collect() để tránh lỗi worker connection
        result_row = predictions.select("prediction").first()
        
        if result_row:
            predicted_aqi = result_row[0]
            aqi_status = get_aqi_status(predicted_aqi)
            st.success(f"Dự đoán chỉ số AQI: {predicted_aqi:.2f} ({aqi_status})")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")