import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import folium_static
import time

# --- 1. AYARLAR VE CİHAZ SEÇİMİ ---
# Klasör adını senin uyarın doğrultusunda tam olarak eşitledik
DATA_ROOT = "./scenario32/" 
MODEL_PATH = "best_multimodal_beam_model.pth"
NUM_BEAMS = 64
LIDAR_POINTS = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MULTI-MODAL MODEL MİMARİSİ (Kaggle Eğitimi ile %100 Uyumlu) ---
class MultiModalBeamPredictor(nn.Module):
    def __init__(self, num_classes=64): 
        super(MultiModalBeamPredictor, self).__init__()
        
        # Image Encoder
        self.image_encoder = models.resnet18(pretrained=False)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 256)
        
        # LiDAR Encoder
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveMaxPool1d(1) 
        )
        self.lidar_fc = nn.Linear(128, 128)
        
        # Radar Encoder
        self.radar_encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        self.radar_fc = nn.Linear(512, 256)
        
        # GPS Encoder
        self.gps_encoder = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        
        # Fusion Katmanı
        self.fusion = nn.Sequential(
            nn.Linear(672, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, image, radar, lidar, gps):
        img_features = self.image_encoder(image)
        lidar_features = self.lidar_fc(self.lidar_encoder(lidar).squeeze(-1))
        
        radar_cnn_out = self.radar_encoder(radar)
        radar_features = self.radar_fc(radar_cnn_out.view(radar_cnn_out.size(0), -1))
        
        gps_features = self.gps_encoder(gps)
        combined_features = torch.cat((img_features, lidar_features, radar_features, gps_features), dim=1)
        return self.fusion(combined_features)

# --- 3. MODEL VE VERİ YÜKLEME YARDIMCILARI ---
@st.cache_resource
def load_model():
    model = MultiModalBeamPredictor(num_classes=NUM_BEAMS).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model
    else:
        st.sidebar.error(f"Model file not found: {MODEL_PATH}. Please copy it into this folder.")
        return None

def read_gps_file(filepath):
    try:
        with open(filepath, 'r') as f:
            coords = f.read().strip().split(',')
            return np.array([float(coords[0]), float(coords[1])], dtype=np.float32)
    except:
        return np.array([33.4242, -111.9281], dtype=np.float32) 

def get_real_powers_normalized(filepath):
    """mmWave pwr dosyasını okur ve maksimum değere bölerek 1'e ORANLAR (Normalizasyon)"""
    try:
        with open(filepath, 'r') as f:
            data = [float(line.strip()) for line in f.readlines() if line.strip()]
        powers = np.array(data[:NUM_BEAMS])
        
        # 🎯 İSTEDİĞİN BİRE ORANLAMA MANTIĞI:
        max_val = np.max(powers)
        if max_val > 0:
            powers = powers / max_val # En büyük güç değerini 1.0 yapar, diğerlerini ona oranlar.
            
        return powers
    except:
        return np.zeros(NUM_BEAMS)

def prepare_inputs(row, base_dir):
    """Her satır için verileri diskten senkronize okur ve hazırlar"""
    # 1. RGB Kamera Resmi
    img_path = os.path.normpath(os.path.join(base_dir, row['unit1_rgb']))
    try:
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        image_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # 2. Radar (.npy)
    radar_path = os.path.normpath(os.path.join(base_dir, row['unit1_radar']))
    try:
        radar_data = np.load(radar_path).astype(np.float32)
    except:
        radar_data = np.zeros((4, 256, 250), dtype=np.float32)
    radar_tensor = torch.tensor(radar_data).unsqueeze(0).to(device)

    # 3. LiDAR (.ply) -> Makaledeki gibi yoğun göstermek için iyileştirildi
    lidar_path = os.path.normpath(os.path.join(base_dir, row['unit1_lidar']))
    try:
        pcd = o3d.io.read_point_cloud(lidar_path)
        points = np.asarray(pcd.points, dtype=np.float32)
    except:
        points = np.zeros((LIDAR_POINTS, 3), dtype=np.float32)
        
    gorsel_points = points.copy() # Ekranda makaledeki gibi yoğun gözükecek ham veri
        
    # Model arka planda yine sabit 1024 nokta ile beslenmeye devam eder
    if len(points) >= LIDAR_POINTS:
        model_points = points[np.random.choice(len(points), LIDAR_POINTS, replace=False)]
    else:
        padding = np.zeros((LIDAR_POINTS - len(points), 3), dtype=np.float32)
        model_points = np.vstack((points, padding))
    lidar_tensor = torch.tensor(model_points).transpose(0, 1).unsqueeze(0).to(device)

    # 4. GPS Verileri
    u1_gps = read_gps_file(os.path.normpath(os.path.join(base_dir, row['unit1_loc'])))
    u2_gps = read_gps_file(os.path.normpath(os.path.join(base_dir, row['unit2_loc'])))
    gps_tensor = torch.tensor(u1_gps, dtype=torch.float32).unsqueeze(0).to(device)

    return image_tensor, radar_tensor, lidar_tensor, gps_tensor, image_rgb, radar_data, gorsel_points, u1_gps, u2_gps

# --- 4. STREAMLIT ARAYÜZ TASARIMI ---
st.set_page_config(page_title="LexiCore Beam Prediction", layout="wide")
st.title("📡 LexiCore: Multi-Modal 6G Beam Prediction Dashboard (Scenario 32)")

# CSV Yükle
csv_full_path = os.path.normpath(os.path.join(DATA_ROOT, "scenario32_dev.csv"))
if not os.path.exists(csv_full_path):
    st.error(f"Error: '{csv_full_path}' not found. Please check the folder name and structure.")
    st.stop()

df = pd.read_csv(csv_full_path)
model = load_model()

# Kontroller (Sidebar)
st.sidebar.header("Simulation Controls")
sim_speed = st.sidebar.slider("Simulation Speed (s/step)", 0.2, 3.0, 0.8)
start_idx = st.sidebar.slider("Start Index (Data Row)", 0, len(df)-1, 0)

if 'running' not in st.session_state:
    st.session_state.running = False
if 'idx' not in st.session_state:
    st.session_state.idx = start_idx

if st.sidebar.button("Start / Stop Simulation"):
    st.session_state.running = not st.session_state.running
    if st.session_state.running:
        st.session_state.idx = start_idx

# Ana Döngü Tetikleyicisi
if model and st.session_state.idx < len(df):
    row = df.iloc[st.session_state.idx]
    
    # Tüm Verileri Çek
    img_t, rad_t, lid_t, gps_t, img_vis, radar_vis, lidar_vis, u1_coords, u2_coords = prepare_inputs(row, DATA_ROOT)
    
    # Yapay Zeka Tahmini
    with torch.no_grad():
        outputs = model(img_t, rad_t, lid_t, gps_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
    top3_indices = np.argsort(probs)[-3:][::-1]
    predicted_beam = top3_indices[0] + 1  
    
    # Gerçek Veriler ve Bire Oranlanmış Güç Grafiği
    real_beam = int(row['unit1_beam'])
    real_powers_normalized = get_real_powers_normalized(os.path.normpath(os.path.join(DATA_ROOT, row['unit1_pwr_60ghz'])))
    
    st.markdown(f"#### ⏱️ Timestamp: `{row['time_stamp']}` | 📊 Current Index: `{st.session_state.idx}`")
    
    # --- ÜST PANEL: SENSÖR GÖRSELLERİ (Yan Yana 3 Kolon) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Kamera (RGB)")
        st.image(img_vis, use_column_width=True)
        
    with col2:
        st.subheader("📟 Radar (Isı Haritası - Kanal 0)")
        fig_radar = px.imshow(radar_vis[0], color_continuous_scale='jet', aspect="auto")
        fig_radar.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col3:
        st.subheader("☁️ LiDAR (3D Nokta Bulutu - Yoğun)")
        # marker size=1 yapılarak noktalar sıklaştırıldı (makale görünümü)
        fig_lidar = go.Figure(data=[go.Scatter3d(
            x=lidar_vis[:,0], y=lidar_vis[:,1], z=lidar_vis[:,2],
            mode='markers', marker=dict(size=1, color=lidar_vis[:,2], colorscale='Viridis')
        )])
        fig_lidar.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
        st.plotly_chart(fig_lidar, use_container_width=True)

    st.markdown("---")
    
    # --- ALT PANEL: SKORLAR VE GRAFİKLER ---
    res_col1, res_col2, res_col3 = st.columns([1.2, 2, 2])
    
    with res_col1:
        st.subheader("🎯 Live Performance")
        st.metric(label="Ground Truth Beam", value=f"Beam {real_beam}")
        st.metric(label="Model Prediction (Top-1)", value=f"Beam {predicted_beam}", 
                  delta="Perfect Match!" if real_beam == predicted_beam else "Mismatch Detected", 
                  delta_color="normal" if real_beam == predicted_beam else "inverse")
        st.write(f"**Alternative Predictions (Top-3):**")
        st.write(f"• 2nd Choice: Beam {top3_indices[1]+1}")
        st.write(f"• 3rd Choice: Beam {top3_indices[2]+1}")
        
    with res_col2:
        st.subheader("🔮 Model Output Probabilities")
        fig_pred = go.Figure(data=[go.Bar(x=np.arange(1, 65), y=probs, marker_color='blue')])
        # y-axis range fixed to 1.1 for visual consistency with normalization
        fig_pred.update_layout(xaxis_title="Beam Index", yaxis_title="Probability (Softmax)", yaxis_range=[0, 1.1], height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pred, use_container_width=True)
        
    with res_col3:
        st.subheader("📶 Normalized Ground Truth Powers")
        fig_real = go.Figure(data=[go.Bar(x=np.arange(1, 65), y=real_powers_normalized, marker_color='red')])
        # y-axis range fixed to 1.1 for visual consistency with normalization
        fig_real.update_layout(xaxis_title="Beam Index", yaxis_title="Normalized Power (Max=1)", yaxis_range=[0, 1.1], height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig_real, use_container_width=True)

    st.markdown("---")
    
    # --- EN ALT PANEL: GPS HARİTASI ---
    st.subheader("🗺️ GPS Geolocation Tracking")
    m = folium.Map(location=[(u1_coords[0] + u2_coords[0]) / 2, (u1_coords[1] + u2_coords[1]) / 2], zoom_start=18)
    
    folium.Marker(location=list(u1_coords), popup="Unit 1 (Receiver Base Station)", 
                  icon=folium.Icon(color='red', icon='broadcast-tower', prefix='fa')).add_to(m)
    folium.Marker(location=list(u2_coords), popup="Unit 2 (Transmitting Vehicle)", 
                  icon=folium.Icon(color='blue', icon='car', prefix='fa')).add_to(m)
    folium.PolyLine(locations=[list(u1_coords), list(u2_coords)], color='green', weight=3, opacity=0.7, tooltip="Signal Line (LOS)").add_to(m)
    
    folium_static(m, width=1100, height=350)

    # Simülasyon Devamı
    if st.session_state.running:
        st.session_state.idx += 1
        time.sleep(sim_speed)
        st.rerun()
else:
    if model:
        st.info("Ready to start the simulation. Click the button in the left menu to begin.")