# 🚀 DeepCSAT – Customer Satisfaction Prediction Using ANN

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square\&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?style=flat-square\&logo=streamlit)

## 📌 Overview

DeepCSAT is an end-to-end Machine Learning project developed to predict Customer Satisfaction (CSAT) using an Artificial Neural Network (ANN). The project covers data preprocessing, feature engineering, model training, evaluation, and deployment through a Streamlit web application for real-time predictions.

---

## 🛠️ Tech Stack

**Python • TensorFlow • Keras • Scikit-Learn • Pandas • NumPy • Matplotlib • Seaborn • Streamlit • Jupyter Notebook • Git • GitHub**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

---

## ✨ Features

* Customer Satisfaction (CSAT) prediction using ANN
* Data preprocessing and feature engineering
* Baseline model comparison
* Model evaluation and performance analysis
* Streamlit-based deployment
* Real-time prediction interface

---

## 📂 Project Structure

```text
DeepCSAT/
│
├── data/
├── models/
│   ├── best_deepcsat_model.keras
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── lr_baseline.pkl
│   └── rf_baseline.pkl
│
├── notebooks/
│   └── DeepCSAT_Capstone_Notebook.ipynb
│
├── outputs/
├── src/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Used

* Logistic Regression (Baseline)
* Random Forest (Baseline)
* Artificial Neural Network (Final Model)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/DeepCSAT.git
cd DeepCSAT

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run src/streamlit_app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## 🔄 Workflow

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
Streamlit Deployment
```


---

## 📈 Results

The Artificial Neural Network outperformed traditional machine learning baseline models and was selected as the final deployment model. The Streamlit application provides an intuitive interface for generating customer satisfaction predictions in real time.

---

## 🚀 Future Enhancements

* Cloud Deployment (AWS/Azure/GCP)
* Automated Retraining Pipeline
* Docker Containerization

---

## 👨‍💻 Author

**Shourya**

Capstone Project – Customer Satisfaction Prediction Using Artificial Neural Networks

---

## 📜 License

This project is developed for academic and educational purposes.
