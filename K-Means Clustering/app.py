"""
K-Means Clustering Web App
รันด้วย: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ================== Page Config ==================
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Custom CSS ==================
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-card h3 {
        color: #667eea;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4f8 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ================== Load Model ==================
@st.cache_resource
def load_model():
    model_path = Path("outputs/kmeans_model.pkl")
    scaler_path = Path("outputs/scaler.pkl")
    features_path = Path("outputs/feature_names.pkl")
    
    if not model_path.exists():
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    return model, scaler, features

model, scaler, feature_names = load_model()

# ================== Header ==================
st.markdown("""
<div class="main-header">
    <h1>🔮 K-Means Clustering App</h1>
    <p>Machine Learning Model Visualization • Iris Dataset</p>
</div>
""", unsafe_allow_html=True)

# ================== Sidebar ==================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("---")
    
    menu = st.radio(
        "เลือกเมนู",
        ["📊 ภาพรวมโมเดล", "🔍 ทำนายข้อมูลใหม่", "📈 วิเคราะห์ Cluster", "ℹ️ เกี่ยวกับ"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 ข้อมูลโมเดล")
    if model is not None:
        st.info(f"""
        - **Algorithm:** K-Means
        - **Clusters:** {model.n_clusters}
        - **Features:** {len(feature_names)}
        - **Iterations:** {model.n_iter_}
        """)
    else:
        st.error("❌ ยังไม่ได้เทรนโมเดล!\nกรุณารัน `python train_model.py` ก่อน")

# ================== Check Model ==================
if model is None:
    st.error("⚠️ ไม่พบโมเดล! กรุณารัน `python train_model.py` ก่อนเพื่อสร้างโมเดล")
    st.stop()

# ================== Menu: ภาพรวม ==================
if menu == "📊 ภาพรวมโมเดล":
    st.markdown("## 📊 ภาพรวมของโมเดล")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>จำนวน Cluster</h3>
            <div class="value">{model.n_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features</h3>
            <div class="value">{len(feature_names)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Inertia</h3>
            <div class="value">{model.inertia_:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Iterations</h3>
            <div class="value">{model.n_iter_}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cluster Centers
    st.markdown("### 🎯 ศูนย์กลางของแต่ละ Cluster (Cluster Centers)")
    centers_df = pd.DataFrame(
        model.cluster_centers_,
        columns=feature_names,
        index=[f"Cluster {i}" for i in range(model.n_clusters)]
    )
    st.dataframe(centers_df.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)
    
    # Load sample data
    sample_df = pd.read_csv("outputs/sample_predictions.csv")
    
    # 2D Visualization
    st.markdown("### 🌌 การกระจายตัวของข้อมูล (2D Visualization)")
    col_a, col_b = st.columns(2)
    x_feature = col_a.selectbox("เลือกแกน X", feature_names, index=0)
    y_feature = col_b.selectbox("เลือกแกน Y", feature_names, index=1)
    
    fig = px.scatter(
        sample_df,
        x=x_feature,
        y=y_feature,
        color=sample_df['cluster'].astype(str),
        title=f"Cluster Distribution: {x_feature} vs {y_feature}",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'cluster': 'Cluster'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample Data Table
    st.markdown("### 📋 ตัวอย่างข้อมูลทำนาย")
    st.dataframe(sample_df, use_container_width=True)

# ================== Menu: ทำนาย ==================
elif menu == "🔍 ทำนายข้อมูลใหม่":
    st.markdown("## 🔍 ทำนายข้อมูลใหม่")
    st.markdown("กรอกค่าของคุณลักษณะทั้ง 4 เพื่อทำนายว่าอยู่ใน Cluster ไหน")
    
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.slider(feature_names[0], 0.0, 10.0, 5.0, 0.1)
        f2 = st.slider(feature_names[1], 0.0, 10.0, 3.0, 0.1)
    with col2:
        f3 = st.slider(feature_names[2], 0.0, 10.0, 3.0, 0.1)
        f4 = st.slider(feature_names[3], 0.0, 10.0, 1.0, 0.1)
    
    if st.button("🚀 ทำนายเลย!", use_container_width=True):
        input_data = np.array([[f1, f2, f3, f4]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        distances = model.transform(input_scaled)[0]
        
        # แสดงผล
        st.markdown("---")
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #48bb78;">
                <h3>Cluster ที่ทำนายได้</h3>
                <div class="value" style="color: #48bb78;">Cluster {prediction}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("#### 📏 ระยะห่างจากแต่ละ Cluster Center")
            dist_df = pd.DataFrame({
                'Cluster': [f"Cluster {i}" for i in range(model.n_clusters)],
                'Distance': distances
            })
            fig = px.bar(
                dist_df,
                x='Cluster',
                y='Distance',
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ================== Menu: วิเคราะห์ ==================
elif menu == "📈 วิเคราะห์ Cluster":
    st.markdown("## 📈 วิเคราะห์ Cluster")
    
    sample_df = pd.read_csv("outputs/sample_predictions.csv")
    
    # จำนวนข้อมูลในแต่ละ Cluster
    st.markdown("### 📊 จำนวนข้อมูลในแต่ละ Cluster")
    cluster_counts = sample_df['cluster'].value_counts().sort_index().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(
            cluster_counts,
            values='Count',
            names='Cluster',
            title='สัดส่วนข้อมูลในแต่ละ Cluster',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            cluster_counts,
            x='Cluster',
            y='Count',
            title='จำนวนข้อมูลในแต่ละ Cluster',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Radar Chart ของ Cluster Centers
    st.markdown("### 🕸️ Radar Chart: ลักษณะของแต่ละ Cluster")
    
    centers_df = pd.DataFrame(model.cluster_centers_, columns=feature_names)
    
    fig_radar = go.Figure()
    colors = px.colors.qualitative.Set2
    
    for i in range(model.n_clusters):
        values = centers_df.iloc[i].tolist()
        values.append(values[0])  # ปิดวงกลม
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=feature_names + [feature_names[0]],
            fill='toself',
            name=f'Cluster {i}',
            opacity=0.5,
            line_color=colors[i % len(colors)]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(255,255,255,0.8)',
            radialaxis=dict(visible=True)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title='ลักษณะเฉพาะของแต่ละ Cluster',
        showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Elbow Image
    st.markdown("### 📉 Elbow Analysis")
    if Path("outputs/elbow_analysis.png").exists():
        st.image("outputs/elbow_analysis.png", use_container_width=True)

# ================== Menu: เกี่ยวกับ ==================
elif menu == "ℹ️ เกี่ยวกับ":
    st.markdown("## ℹ️ เกี่ยวกับแอปพลิเคชัน")
    
    st.markdown("""
    ### 🎯 วัตถุประสงค์
    แอปพลิเคชันนี้แสดงผลลัพธ์ของโมเดล **K-Means Clustering** 
    ที่เทรนบน **Iris Dataset** เพื่อจัดกลุ่มดอกไม้ตามลักษณะ 4 ประการ
    
    ### 🛠️ เทคโนโลยีที่ใช้
    | ส่วน | เทคโนโลยี |
    |------|-----------|
    | Machine Learning | scikit-learn |
    | Data Processing | pandas, numpy |
    | Visualization | plotly, matplotlib |
    | Web Framework | Streamlit |
    
    ### 📋 ขั้นตอนการทำงาน
    1. **Data Preprocessing** - ตรวจสอบ missing values และ outliers
    2. **Train/Test Split** - แบ่งข้อมูล 80/20
    3. **Transform** - StandardScaler เพื่อ normalize ข้อมูล
    4. **Elbow Method** - หาจำนวน Cluster ที่เหมาะสม
    5. **K-Means Training** - เทรนโมเดล
    6. **Evaluation** - ประเมินด้วย Silhouette Score
    
    ### 👨‍💻 ผู้พัฒนา
    สร้างด้วย ❤️ โดย AI Assistant
    """)

# ================== Footer ==================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 1rem;">
    <p>🔮 K-Means Clustering App • Built with Streamlit & scikit-learn</p>
</div>
""", unsafe_allow_html=True)