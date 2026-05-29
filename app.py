# ============================================================
#  K-MEANS CUSTOMER SEGMENTATION
#  SkillCraft Technology — Task 02
#  Author : MAYANK
# ============================================================


# ─────────────────────────────────────────────────────────────
# STEP 1 : Download Dataset
# ─────────────────────────────────────────────────────────────

import urllib.request
import os

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/mall_customers.csv"
filename = "Mall_Customers.csv"

if not os.path.exists(filename):

    try:
        urllib.request.urlretrieve(url, filename)
        print("✅ Dataset downloaded successfully!")

    except:
        print("⚠️ Download failed. Creating synthetic dataset...")

        import numpy as np
        import pandas as pd

        np.random.seed(42)

        n = 200

        synthetic_data = {
            "CustomerID": range(1, n + 1),

            "Genre": np.random.choice(
                ["Male", "Female"], n
            ),

            "Age": np.random.randint(
                18, 70, n
            ),

            "Annual Income (k$)": np.concatenate([
                np.random.normal(80, 8, 40),
                np.random.normal(80, 8, 40),
                np.random.normal(50, 8, 40),
                np.random.normal(25, 8, 40),
                np.random.normal(25, 8, 40)
            ]).clip(15, 140).astype(int),

            "Spending Score (1-100)": np.concatenate([
                np.random.normal(82, 8, 40),
                np.random.normal(18, 8, 40),
                np.random.normal(50, 10, 40),
                np.random.normal(79, 8, 40),
                np.random.normal(20, 8, 40)
            ]).clip(1, 100).astype(int)
        }

        df_temp = pd.DataFrame(synthetic_data)
        df_temp.to_csv(filename, index=False)

        print("✅ Synthetic dataset created successfully!")

else:
    print("✅ Dataset already available!")


# ─────────────────────────────────────────────────────────────
# STEP 2 : Import Required Libraries
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

print("✅ Libraries imported successfully!")


# ─────────────────────────────────────────────────────────────
# STEP 3 : Load Dataset
# ─────────────────────────────────────────────────────────────

df = pd.read_csv(filename)

df.columns = df.columns.str.strip()

print("\n" + "="*50)
print("DATASET INFORMATION")
print("="*50)

print(f"Dataset Shape : {df.shape}")

print("\nFirst 5 Rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe().round(2))

print("\nMissing Values:")
print(df.isnull().sum())


# ─────────────────────────────────────────────────────────────
# STEP 4 : Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")

# Age Distribution
axes[0].hist(
    df["Age"],
    bins=20,
    color="#58a6ff",
    edgecolor="black"
)

axes[0].set_title(
    "Age Distribution",
    color="white"
)

axes[0].tick_params(colors="white")


# Income Distribution
axes[1].hist(
    df["Annual Income (k$)"],
    bins=20,
    color="#3fb950",
    edgecolor="black"
)

axes[1].set_title(
    "Annual Income Distribution",
    color="white"
)

axes[1].tick_params(colors="white")


# Spending Score Distribution
axes[2].hist(
    df["Spending Score (1-100)"],
    bins=20,
    color="#f78166",
    edgecolor="black"
)

axes[2].set_title(
    "Spending Score Distribution",
    color="white"
)

axes[2].tick_params(colors="white")

plt.tight_layout()

plt.savefig(
    "eda_plots.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0d1117"
)

plt.show()

print("✅ EDA plot saved successfully!")


# ─────────────────────────────────────────────────────────────
# STEP 5 : Select Features
# ─────────────────────────────────────────────────────────────

X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].values

print("\nSelected Features:")
print("1. Annual Income")
print("2. Spending Score")


# ─────────────────────────────────────────────────────────────
# STEP 6 : Elbow Method & Silhouette Score
# ─────────────────────────────────────────────────────────────

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []

K_values = range(2, 11)

print("\nSearching for best K value...\n")

for k in K_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

    score = silhouette_score(
        X_scaled,
        model.labels_
    )

    silhouette_scores.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette = {score:.4f}"
    )

best_k = list(K_values)[
    np.argmax(silhouette_scores)
]

print(f"\n✅ Best K value = {best_k}")


# ─────────────────────────────────────────────────────────────
# STEP 7 : Plot Elbow & Silhouette Graph
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")

# Elbow Plot
axes[0].plot(
    K_values,
    inertias,
    marker="o",
    linewidth=2,
    color="#58a6ff"
)

axes[0].set_title(
    "Elbow Method",
    color="white"
)

axes[0].set_xlabel(
    "K Value",
    color="white"
)

axes[0].set_ylabel(
    "Inertia",
    color="white"
)

axes[0].tick_params(colors="white")


# Silhouette Plot
axes[1].plot(
    K_values,
    silhouette_scores,
    marker="o",
    linewidth=2,
    color="#f78166"
)

axes[1].axvline(
    x=best_k,
    linestyle="--",
    color="#3fb950",
    label=f"Best K = {best_k}"
)

axes[1].legend()

axes[1].set_title(
    "Silhouette Score",
    color="white"
)

axes[1].set_xlabel(
    "K Value",
    color="white"
)

axes[1].set_ylabel(
    "Score",
    color="white"
)

axes[1].tick_params(colors="white")

plt.tight_layout()

plt.savefig(
    "elbow_silhouette.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0d1117"
)

plt.show()

print("✅ Elbow & Silhouette graph saved!")


# ─────────────────────────────────────────────────────────────
# STEP 8 : Apply Final K-Means Clustering
# ─────────────────────────────────────────────────────────────

K = 5

kmeans = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=10,
    random_state=42
)

labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)

print("\n" + "="*50)
print(f"K-MEANS RESULTS (K = {K})")
print("="*50)

print(f"Inertia Score      : {kmeans.inertia_:.2f}")
print(f"Silhouette Score   : {score:.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 9 : Cluster Visualization
# ─────────────────────────────────────────────────────────────

colors = [
    "#f78166",
    "#58a6ff",
    "#3fb950",
    "#d2a8ff",
    "#ffa657"
]

fig, ax = plt.subplots(figsize=(10, 7))

fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

for i in range(K):

    cluster_points = labels == i

    ax.scatter(
        X[cluster_points, 0],
        X[cluster_points, 1],
        s=70,
        color=colors[i],
        label=f"Cluster {i+1}",
        alpha=0.8
    )

# Plot Centroids
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="*",
    s=300,
    color="white",
    edgecolors="black",
    label="Centroids"
)

ax.set_title(
    "K-Means Customer Segmentation",
    color="white",
    fontsize=14
)

ax.set_xlabel(
    "Annual Income (k$)",
    color="white"
)

ax.set_ylabel(
    "Spending Score (1-100)",
    color="white"
)

ax.tick_params(colors="white")

ax.legend()

plt.tight_layout()

plt.savefig(
    "clusters_k5.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0d1117"
)

plt.show()

print("✅ Cluster visualization saved!")


# ─────────────────────────────────────────────────────────────
# STEP 10 : Cluster Analysis
# ─────────────────────────────────────────────────────────────

df["Cluster"] = labels + 1

summary = df.groupby("Cluster").agg({

    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean",
    "CustomerID": "count"

}).round(1)

summary.rename(columns={
    "CustomerID": "Total Customers"
}, inplace=True)

print("\n" + "="*60)
print("CLUSTER ANALYSIS REPORT")
print("="*60)

print(summary)

# Save Final Dataset
df.to_csv(
    "customers_clustered.csv",
    index=False
)

print("\n✅ Clustered dataset saved successfully!")


# ─────────────────────────────────────────────────────────────
# STEP 11 : Cluster Summary Chart
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")

x = np.arange(K)

cluster_names = [
    f"C{i+1}" for i in range(K)
]

# Income Bar Chart
bars1 = axes[0].bar(
    x,
    summary["Annual Income (k$)"],
    color=colors
)

axes[0].set_title(
    "Average Income per Cluster",
    color="white"
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(cluster_names)

axes[0].tick_params(colors="white")


# Spending Score Bar Chart
bars2 = axes[1].bar(
    x,
    summary["Spending Score (1-100)"],
    color=colors
)

axes[1].set_title(
    "Average Spending Score",
    color="white"
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(cluster_names)

axes[1].tick_params(colors="white")

plt.tight_layout()

plt.savefig(
    "cluster_summary.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0d1117"
)

plt.show()

print("✅ Summary chart saved!")


# ─────────────────────────────────────────────────────────────
# FINAL OUTPUT
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("="*60)

print(f"Dataset Rows     : {df.shape[0]}")
print(f"Number of Clusters : {K}")
print(f"Silhouette Score : {score:.4f}")

print("\nGenerated Files:")
print("1. eda_plots.png")
print("2. elbow_silhouette.png")
print("3. clusters_k5.png")
print("4. cluster_summary.png")
print("5. customers_clustered.csv")

print("\n✅ Project completed successfully!")
print("="*60)
