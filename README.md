# MQTT IoT Intrusion Detection System (MQTT-IDS)

Hệ thống phát hiện xâm nhập (Intrusion Detection System - IDS) cho mạng IoT giao thức MQTT dựa trên Machine Learning & Deep Learning, được xây dựng trên bộ dữ liệu **MQTT Internet of Things Intrusion Detection Dataset** (Hanan Hindy et al., IEEE DataPort 2020).

---

## 1. Cấu Trúc Dữ Liệu

Bộ dữ liệu cung cấp 3 định dạng biểu diễn lưu lượng mạng:
1. **Biflow (`biflow_*.csv`)** *(Khuyến nghị & Tối ưu nhất)*: Lưu lượng hai chiều (Forward & Backward) với 28 đặc trưng mạng (IAT statistics, packet length, flags, bytes). Tổng cộng **157,009 mẫu**.
2. **Uniflow (`uniflow_*.csv`)**: Lưu lượng một chiều với 15 đặc trưng thống kê luồng.
3. **Packet-level (`*.csv` & `*.pcap`)**: Chi tiết từng gói tin thô ở tầng mạng và tầng ứng dụng MQTT.

### Các Lớp Phân Loại (5 Classes):
- **0 - Normal**: Lưu lượng MQTT thông thường giữa các thiết bị cảm biến IoT và MQTT broker.
- **1 - Scan_A**: Tấn công quét cổng Nmap Aggressive / SYN Scan.
- **2 - Scan_sU**: Tấn công quét cổng UDP (UDP Scan).
- **3 - Sparta**: Công cụ tự động Sparta dò mật khẩu / brute-force qua mạng.
- **4 - MQTT_Bruteforce**: Tấn công brute-force từ điển nhắm trực tiếp vào dịch vụ MQTT (port 1883).

---

## 2. Các File Chính Trong Project

- [`mqtt_dataset.py`](file:///d:/mqtt/mqtt_dataset.py): Module tiền xử lý dữ liệu, chuẩn hóa đặc trưng (`StandardScaler`), xử lý giá trị khuyết/vô hạn, tách tập train/test có phân tầng (`stratify`), và loại bỏ triệt để rò rỉ nhãn (`target leakage`).
- [`train_ml_models.py`](file:///d:/mqtt/train_ml_models.py): Pipeline huấn luyện mô hình Machine Learning cổ điển: **Random Forest**, **HistGradientBoosting** (LightGBM-style), và **Decision Tree**.
- [`train_deep_model.py`](file:///d:/mqtt/train_deep_model.py): Mô hình Deep Learning PyTorch (`MQTTNet`) tối ưu hóa cho mạng nơ-ron đa tầng với BatchNorm, Dropout, hỗ trợ GPU (NVIDIA RTX 2050 CUDA).
- [`predict_mqtt.py`](file:///d:/mqtt/predict_mqtt.py): Script thực thi suy luận (Inference), nhận đầu vào luồng mạng và dự đoán loại tấn công kèm độ tin cậy (% confidence).

---

## 3. Kết Quả Huấn Luyện (Benchmark Results)

Đánh giá trên tập kiểm thử độc lập (31,402 mẫu kiểm thử `biflow`):

| Mô Hình | Accuracy | Balanced Accuracy | Macro F1-Score | Weighted F1-Score | Thời Gian Train |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **99.97%** | **99.97%** | **99.97%** | **99.97%** | 1.72s |
| **HistGradientBoosting** | **99.96%** | **99.96%** | **99.95%** | **99.96%** | 2.41s |
| **Decision Tree** | **99.96%** | **99.96%** | **99.96%** | **99.96%** | 0.66s |
| **MQTTNet (Deep Learning PyTorch)** | **99.59%** | **99.66%** | **99.51%** | **99.59%** | 33.34s (GPU) |

*Tất cả biểu đồ Ma trận nhầm lẫn (Confusion Matrix) và Feature Importance đã được xuất vào thư mục [`reports/`](file:///d:/mqtt/reports).*

---

## 4. Hướng Dẫn Chạy & Sử Dụng

### 4.1. Chạy Dự Đoán / Kiểm Tra Mô Hình (Inference)
Dự đoán bằng mô hình Random Forest tốt nhất:
```bash
python predict_mqtt.py --model_path models/best_mqtt_model_biflow_multiclass.joblib
```

Dự đoán bằng mô hình Deep Learning PyTorch:
```bash
python predict_mqtt.py --model_path models/mqttnet_biflow_multiclass.pt
```

Dự đoán từ một file CSV chứa các luồng mạng mới:
```bash
python predict_mqtt.py --model_path models/best_mqtt_model_biflow_multiclass.joblib --data_file path/to/your_test_traffic.csv
```

### 4.2. Huấn Luyện Lại Mô Hình Machine Learning
```bash
# Huấn luyện toàn bộ mô hình trên biflow (5 lớp)
python train_ml_models.py --mode biflow --classification multiclass --model all

# Huấn luyện phân loại nhị phân (Normal vs Attack)
python train_ml_models.py --mode biflow --classification binary --model rf
```

### 4.3. Huấn Luyện Lại Mô Hình Deep Learning (PyTorch GPU)
```bash
python train_deep_model.py --mode biflow --classification multiclass --epochs 15 --batch_size 256
```
