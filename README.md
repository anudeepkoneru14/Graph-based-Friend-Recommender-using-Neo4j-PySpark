# 🤝 Friend Recommendation System using Neo4j, PySpark & Streamlit

A full-stack machine learning project that predicts potential friend connections using graph data from Neo4j and builds a real-time interactive dashboard using Streamlit.

---

## 📌 Project Overview

This project demonstrates a graph-based link prediction system:
- Graph data modeled in **Neo4j**
- Features engineered and trained using **PySpark MLlib**
- Recommendations visualized through a **Streamlit dashboard**

---

## 🚀 Tech Stack

- **Neo4j**: Graph database for storing and querying user relationships
- **PySpark**: ML pipeline for binary classification (friend vs non-friend)
- **Pandas**: Preprocessing and data manipulation
- **Streamlit**: Interactive frontend dashboard
- **Jupyter Notebook**: Development environment

---

## 📊 Dataset

Synthetic data generated in Neo4j using Cypher and APOC:
- `User` nodes with attributes: `id`, `name`, `age`, `location`, `interests`
- `FRIEND` relationships with optional properties

Total:
- 200 users
- ~500 FRIEND relationships
- Negative samples generated to balance the dataset

---

## 🧠 ML Pipeline

1. **Data Export** from Neo4j using Python driver
2. **Negative Sample Generation** to create non-friend pairs
3. **Feature Engineering**:
   - StringIndexer for `user1` and `user2`
   - VectorAssembler to combine features
4. **Binary Classification** using Logistic Regression
5. **Prediction Output** with probabilities of forming a friendship

---

## 🖥️ Streamlit Dashboard

Interactive app allows:
- Selection of a user
- Viewing top friend recommendations sorted by confidence score
- (Optional) Data visualization using bar charts

---

## 📁 File Structure

