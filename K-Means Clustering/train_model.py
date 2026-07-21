"""
K-Means Clustering with Data Preprocessing
รันบน CMD: python train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ================== 1. โหลดข้อมูล ==================
print("=" * 50)
print("📊 กำลังโหลดข้อมูล Iris Dataset...")
print("=" * 50)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(f"✅ ขนาดข้อมูล: {df.shape[0]} rows, {df.shape[1]} columns\n")

# ================== 2. Data Preprocessing ==================
print("🔧 กำลังทำ Data Preprocessing...")

# ตรวจสอบ Missing Values
missing = df.isnull().sum().sum()
print(f"   - Missing values: {missing}")

# ตรวจสอบ Outliers ด้วย IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum().sum()
print(f"   - Outliers detected: {outliers}")

# ================== 3. แบ่ง Train / Test ==================
print("\n📌 กำลังแบ่งข้อมูล Train/Test (80/20)...")
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
print(f"   - Train set: {X_train.shape[0]} samples")
print(f"   - Test set : {X_test.shape[0]} samples")

# ================== 4. Transform Data (StandardScaler) ==================
print("\n⚙️  กำลัง Transform ข้อมูลด้วย StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Transform เสร็จสิ้น!")

# ================== 5. หาจำนวน Cluster ที่เหมาะสม (Elbow Method) ==================
print("\n📈 กำลังวิเคราะห์จำนวน Cluster ที่เหมาะสม...")
inertia_values = []
silhouette_values = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(X_train_scaled, kmeans.labels_))

# เลือก K ที่ดีที่สุดจาก Silhouette Score
best_k = list(K_range)[np.argmax(silhouette_values)]
print(f"   🏆 จำนวน Cluster ที่เหมาะสมที่สุด: K = {best_k}")

# บันทึกกราฟ Elbow
os.makedirs("outputs", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(K_range, inertia_values, marker='o', color='#4C72B0')
ax[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Number of Clusters (K)')
ax[0].set_ylabel('Inertia')
ax[0].grid(True, alpha=0.3)

ax[1].plot(K_range, silhouette_values, marker='s', color='#DD8452')
ax[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
ax[1].set_xlabel('Number of Clusters (K)')
ax[1].set_ylabel('Silhouette Score')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/elbow_analysis.png", dpi=150, bbox_inches='tight')
print("   💾 บันทึกกราฟ: outputs/elbow_analysis.png")

# ================== 6. เทรนโมเดลด้วย K ที่เหมาะสม ==================
print(f"\n🚀 กำลังเทรน K-Means ด้วย K = {best_k}...")
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_model.fit(X_train_scaled)

# Predict บน Test set
test_labels = final_model.predict(X_test_scaled)
train_labels = final_model.labels_

# ประเมินผล
train_sil = silhouette_score(X_train_scaled, train_labels)
test_sil = silhouette_score(X_test_scaled, test_labels)
print(f"\n📊 ผลการประเมิน:")
print(f"   - Train Silhouette Score: {train_sil:.4f}")
print(f"   - Test  Silhouette Score: {test_sil:.4f}")

# ================== 7. บันทึกโมเดล ==================
print("\n💾 กำลังบันทึกโมเดลและ scaler...")
joblib.dump(final_model, "outputs/kmeans_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump(list(iris.feature_names), "outputs/feature_names.pkl")

# บันทึกข้อมูลตัวอย่างสำหรับ Streamlit
sample_data = pd.DataFrame(X_test_scaled, columns=iris.feature_names)
sample_data['cluster'] = test_labels
sample_data.to_csv("outputs/sample_predictions.csv", index=False)

print("=" * 50)
print("✅ เสร็จสิ้น! ไฟล์ที่บันทึกในโฟลเดอร์ 'outputs/'")
print("   - kmeans_model.pkl")
print("   - scaler.pkl")
print("   - feature_names.pkl")
print("   - sample_predictions.csv")
print("   - elbow_analysis.png")
print("=" * 50)
print("\n👉 รัน Streamlit ด้วยคำสั่ง: streamlit run app.py")