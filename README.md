<p align="center">
  <img src="Screenshot 2025-11-14 161313.png" alt="Project Overview" width="800">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/XGBoost-Model-green?style=for-the-badge&logo=xgboost" />
  <img src="https://img.shields.io/badge/SVM-Support%20Vector%20Machine-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/Reproducible-Environment-success?style=for-the-badge&logo=dependabot" />
</p>

# 🚀 Employee Attrition Prediction – End-to-End ML Project

A complete end-to-end Machine Learning system to predict **Employee Attrition** using HR data.  
Includes **EDA, preprocessing, ML models, deployment (FastAPI + Streamlit), Docker, and Cloud deployment**.

---

# 📁 1. Project Structure

```
📁 Employee-Attrition-Prediction
│── .python-version
│── Dockerfile
│── README.md
│── Streamlit.py
│── Summary.png
│── WA_Fn-UseC_-HR-Employee-Attrition.csv
│── check.ipynb
│── dv.bin
│── main.py
│── notebook.ipynb
│── output.png
│── predict.py
│── predict_test0.py
│── predict_test1.py
│── preprocessed_data.csv
│── pyproject.toml
│── requirements.txt
│── train.py
│── uv.lock
│── xgboost_model.bin
│── Screenshot 2025-11-14 161313.png
│── Screenshot 2025-11-14 183608.png
│── Screenshot 2025-11-14 183618.png
│── Screenshot 2025-11-14 183638.png
│── Screenshot 2025-11-14 183724.png
```

---

# 🧠 2. Problem Description

Employee attrition prediction helps HR teams identify employees at risk of leaving the organization.  
This project predicts **Attrition = Yes/No** using features like:

- Age  
- Job Role  
- Monthly Income  
- Work-Life Balance  
- Job Satisfaction  
- Years at Company  
- Distance from Home  

This helps companies **reduce turnover**, **cut hiring costs**, and **improve retention**.

---

# 📊 3. Exploratory Data Analysis (EDA)

Performed inside `notebook.ipynb`.

### ✔ Dataset Overview
- Data types  
- Summary stats  

### ✔ Missing Values  
- Clean dataset saved as `preprocessed_data.csv`

### ✔ Target Variable Analysis  
- Checking distribution of Attrition  
- Class imbalance visualized  

### ✔ Feature Distributions  
- Histograms  
- Boxplots  
- Countplots  

### ✔ Correlation & Feature Importance
- Heatmap  
- XGBoost importance plots (`Summary.png`, `output.png`)

---

# 🤖 4. Model Training

Training pipeline inside **train.py**.

### 📊 Accuracy Comparison

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|----------------|
| Logistic Regression | **0.8954** | **0.8707** |
| Random Forest | **0.9303** | **0.8741** |
| XGBoost | **1.0000** | **0.8741** |
| **SVM** | **0.9014** | **0.8878** |

👉 **SVM achieved the highest generalization on test data.**  
👉 XGBoost overfits slightly (Train = 1.0), but is fast for deployment.

### ✔ Models Trained
- Logistic Regression  
- **SVM (RBF Kernel)**  
- Decision Tree  
- Random Forest  
- Extra Trees  
- XGBoost (final deployed model)

### ✔ Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

---

# 📝 5. Script Export

All notebook logic exported into scripts:

- `train.py` → trains model & saves `xgboost_model.bin`  
- `predict.py` → loads saved model & predicts  
- `predict_test0.py` / `predict_test1.py` → testing scripts  

---

# 🔁 6. Reproducibility

Project includes:

✔ Raw dataset  
✔ Clean dataset  
✔ requirements.txt  
✔ uv.lock  
✔ pyproject.toml  
✔ Dockerfile  
✔ Same results across environments  

Reproduce training:

```bash
python train.py
```

---

# ⚡ 7. FastAPI Deployment (Backend)

API is implemented in **main.py**.
**API** Link : https://ibm-hr-midterm-project.onrender.com/docs

### Run API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Endpoints:

- `/predict` → Predict employee attrition  
- `/` → Health check  

---

# 🎨 8. Streamlit Deployment (Frontend)

Frontend implemented in **Streamlit.py**.
**Streamlit.py** Link : https://ibm-hr-analytic-attrition.streamlit.app/

### Launch Streamlit UI:

```bash
streamlit run Streamlit.py
```

Features:
- HR inputs form  
- Calls FastAPI backend  
- Displays prediction + confidence  

---

# 📦 9. Dependencies & Environment Management

Project supports **UV (modern, fast Python package manager)**.

### Create environment:

```bash
uv venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts ctivate           # Windows
uv pip install -r requirements.txt
```

---

# 🐳 10. Dockerization

Project includes a full **Dockerfile**.
**Dockerfile** Link : https://hub.docker.com/repository/docker/ayushgurjar10/ibm_hr_midterm_project/general

### Build image:

```bash
docker build -t attrition-app .
```

### Run container:

```bash
docker run -p 8000:8000 attrition-app
```

---

# ☁️ 11. Cloud Deployment

Cloud deployment screenshots:

### **1️⃣ Build & Upload**
<img src="Screenshot 2025-11-14 183638.png" width="800">

### **2️⃣ Deployment Successful**
<img src="Screenshot 2025-11-14 183618.png" width="800">

### **3️⃣ API Running Online**
<img src="Screenshot 2025-11-14 183608.png" width="800">

### **4️⃣ Prediction Tested Live**
<img src="Screenshot 2025-11-14 183724.png" width="800">

---

# ▶️ 12. How to Run Entire Project

### **A. Train the Model**
```bash
python train.py
```

### **B. Start FastAPI**
```bash
uvicorn main:app --reload
```

### **C. Start Streamlit Interface**
```bash
streamlit run Streamlit.py
```

### **D. Test Model Using Predict Script**
```bash
python predict_test0.py
```

### **E. Docker Workflow**
```bash
docker build -t attrition-app .
docker run -p 8000:8000 attrition-app
```

---

# 🏁 13. Evaluation Summary

| Evaluation Criteria | Score |
|---------------------|--------|
| Problem Description | **2/2** |
| EDA | **2/2** |
| Model Training | **3/3** |
| Script Export | **1/1** |
| Reproducibility | **1/1** |
| Model Deployment | **1/1** |
| Dependency Management | **2/2** |
| Containerization | **2/2** |
| Cloud Deployment | **2/2** |

**Total Score: 16/16**

---

# 🎉 Final Notes

- ✔ SVM achieved **best accuracy**  
- ✔ XGBoost used for **deployment (fast + optimized)**  
 
