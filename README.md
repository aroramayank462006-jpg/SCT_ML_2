# 🛍️ Customer Segmentation Dashboard

## Unsupervised Machine Learning with K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A portfolio-grade Machine Learning web application that segments mall customers into meaningful groups using K-Means Clustering. Built with Python and Streamlit for a clean, professional dashboard experience.

---

# 📸 Features at a Glance

| Feature                  | Description                                         |
| ------------------------ | --------------------------------------------------- |
| 📊 Overview Dashboard    | Dataset summary, gender & age distributions         |
| 🔍 Dataset Preview       | Interactive table, statistics, missing-value check  |
| ⚙️ Preprocessing         | StandardScaler feature scaling, correlation heatmap |
| 📈 Elbow Method          | Interactive chart to find optimal K                 |
| 🤖 Model Training        | KMeans with silhouette score evaluation             |
| 🎯 Cluster Visualisation | 2-D & 3-D scatter plots, distribution chart         |
| 💡 Business Insights     | Segment profiles and marketing recommendations      |
| 📥 Downloads             | Clustered CSV + trained `.pkl` model                |

---

# 🧠 How K-Means Clustering Works

1. **Initialise** — choose K cluster centroids at random
2. **Assign** — place each customer in the nearest centroid's cluster
3. **Update** — recompute centroids as the mean of all assigned points
4. Repeat steps 2–3 until centroids stop moving (convergence)

The algorithm minimises within-cluster sum of squares (WCSS / Inertia).

---

# 📐 The Elbow Method

Run K-Means for `k = 2 … 10` and plot inertia for each k.

The curve bends at the **"elbow"** — the point where adding more clusters yields diminishing returns.

That value of `k` is considered the optimal number of clusters.

---

# ⚖️ Why Feature Scaling?

K-Means relies on Euclidean distance. Without scaling, a feature like **Annual Income** would dominate **Spending Score**.

`StandardScaler` transforms every feature to:

* Mean = 0
* Standard Deviation = 1

This ensures equal contribution from all features.

---

# 📁 Project Structure

```bash
customer_segmentation/
├── app.py
├── mall_customers.csv
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/nikhildembla01/SCT_ML_2.git
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

# 📊 Dataset

The Mall Customer Dataset contains 200 records with these columns:

| Column                 | Type | Description                            |
| ---------------------- | ---- | -------------------------------------- |
| CustomerID             | int  | Unique customer identifier             |
| Gender                 | str  | Male / Female                          |
| Age                    | int  | Customer age (18–70)                   |
| Annual Income (k$)     | int  | Annual income in thousands             |
| Spending Score (1-100) | int  | Mall-assigned spending behaviour score |

---

# 🎯 Customer Segments (5 Clusters)

| Segment              | Profile                           | Strategy                          |
| -------------------- | --------------------------------- | --------------------------------- |
| 🛒 Budget Shoppers   | Young, low income, high spend     | Trend & impulse campaigns         |
| 💎 Premium Loyalists | Mid-age, high income, high spend  | Loyalty rewards, premium launches |
| 💼 Cautious Wealthy  | Older, high income, low spend     | Quality messaging, trust building |
| 🌱 Frugal Starters   | Young, low income, low spend      | Value bundles, growth targeting   |
| ⚖️ Balanced Spenders | Mid-career, medium income & spend | Consistent engagement programmes  |

---

# 🛠️ Tech Stack

* Python 3.10+ — Core language
* Streamlit — Web dashboard framework
* scikit-learn — K-Means, StandardScaler
* Pandas / NumPy — Data manipulation
* Matplotlib / Seaborn — Visualisations
* Pickle — Model persistence

---

# 🎯 Project Objective

The goal of this project is to analyse customer behaviour and divide customers into different groups for better marketing strategies and business decision-making.

---

# 👨‍💻 Internship Task

SkillCraft Technology — Machine Learning Internship Task 02

---

# 📄 License

MIT © 2024 — Free to use and modify.

---
