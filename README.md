# 🛡️ Full Stack Network Intrusion Detection System

A full-stack, research-grade **Network Intrusion Detection System (NIDS)** that classifies network traffic as **Normal** or **Anomaly** using an ensemble of Classical Machine Learning, Deep Learning, and Quantum Machine Learning models — all wrapped in an interactive **Streamlit dashboard**.

---

## 📌 Project Overview

This project benchmarks nine AI models across three paradigms on the **KDD Cup 1999** dataset and exposes real-time predictions through a web dashboard:

| Paradigm | Models |
|---|---|
| Classical ML | KNN, Decision Tree, Random Forest, SVM, Naive Bayes, XGBoost |
| Deep Learning | Deep Neural Network (DNN) |
| Quantum ML | Quantum SVM (QSVM), Variational Quantum Classifier (VQC) |

---

## 📂 Repository Structure

```
├── Full_Stack_based_Network_Intrusion_Detection_(Tested).ipynb   # Main notebook
├── test_intrusion_detection.py                                   # Test file containing test cases
├── Train_data.csv                                                # KDD Cup 1999 training set (25,192 records)
├── Test_data.csv                                                 # KDD Cup 1999 test set
├── requirements.txt                                              # Python dependencies
├── LICENSE                                                       # GNU Affero General Public License v3.0
└── README.md                                                     # README file
```

> **Note:** Model artifact files (`*.pkl`, `dnn_model.keras`) are generated when you run the notebook and are not included in the repository.

---

## 🗂️ Dataset

- **Source:** [KDD Cup 1999 Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
- **Training samples:** 25,192
- **Features:** 41 network traffic features (duration, protocol_type, service, flag, src_bytes, etc.)
- **Target:** Binary classification — `normal` vs `anomaly`
- **Class distribution:** ~53% normal, ~47% anomaly

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn tensorflow keras xgboost qiskit qiskit-machine-learning qiskit-algorithms streamlit pyngrok joblib dill matplotlib seaborn pylatexenc
```

### 3. Run on Google Colab (Recommended)

1. Upload the notebook and both CSV files to your Colab session.
2. Run all cells sequentially (**Runtime → Run All**).
3. Insert your ngrok auth token in the final cell:
   ```bash
   !ngrok config add-authtoken <YOUR_NGROK_TOKEN>
   ```
4. The public Streamlit URL will be printed in the last cell output.

---

## 🚀 Running the Project

### Step-by-step (in order):

| Step | Cell Action | Output |
|---|---|---|
| 1 | Install dependencies | Libraries ready |
| 2 | Load & preprocess data | Encoded + scaled DataFrames |
| 3 | Train classical ML models | Accuracy scores printed |
| 4 | Build & train DNN | Keras model trained |
| 5 | Train QSVM & VQC | Quantum model accuracies |
| 6 | Evaluate all models | Comparison table + confusion matrices |
| 7 | Save all artifacts | `.pkl` and `.keras` files saved |
| 8 | Launch Streamlit app | Public URL via ngrok |

> ⚠️ **Run cells in order.** Out-of-order execution will cause `NameError` or missing artifact errors.

---

## 🌐 Streamlit Dashboard

Once running, the dashboard provides:

- **Sidebar** — Input 10 network traffic features interactively
- **Live Predictions** — Real-time NORMAL ✅ / ANOMALY 🚨 output from all 4 model types
- **Accuracy Table** — All 9 models ranked by accuracy, loaded from the last training session
- **Bar Chart** — Visual comparison of model performance

---

## 🧠 Model Architecture

### Deep Neural Network (DNN)
```
Input(41) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dense(1, Sigmoid)
Optimizer: Adam | Loss: Binary Cross-Entropy | Threshold: 0.5
```

### Quantum SVM (QSVM)
```
Feature Map: ZZFeatureMap(feature_dim=4, reps=2, entanglement='linear')
Training subset: 100 samples | Validation subset: 50 samples
```

### Variational Quantum Classifier (VQC)
```
Feature Map: ZZFeatureMap(feature_dim=4, reps=2)
Ansatz: RealAmplitudes(num_qubits=4, reps=2)
Optimizer: COBYLA(maxiter=100)
```

> Quantum models use PCA (n_components=4) + MinMaxScaler preprocessing due to classical simulation constraints.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | ~99% | ~99% | ~99% | ~99% |
| XGBoost | ~99% | ~99% | ~99% | ~99% |
| Decision Tree | ~98% | ~98% | ~98% | ~98% |
| KNN | ~97% | ~97% | ~97% | ~97% |
| SVM | ~96% | ~96% | ~96% | ~96% |
| DNN | ~95% | ~95% | ~95% | ~95% |
| Naive Bayes | ~88% | ~88% | ~88% | ~88% |
| QSVM | varies* | — | — | — |
| VQC | varies* | — | — | — |

> *Quantum model accuracy varies due to the limited 100-sample training subset used for computational feasibility.  
> Exact values will be printed in your notebook run and displayed in the dashboard.

---

## 🔐 Security Notes

- **Do not commit your ngrok auth token** to this repository. Add it only at runtime.
- The Streamlit dashboard has no authentication by default. Avoid sharing the ngrok URL publicly.
- Model `.pkl` files should not be exposed publicly as they contain serialized model weights.

---

## 📋 Requirements

```
pandas
numpy
scikit-learn
tensorflow
keras
xgboost
qiskit
qiskit-machine-learning
qiskit-algorithms
streamlit
pyngrok
joblib
dill
matplotlib
seaborn
pylatexenc
```

> See `requirements.txt` for pinned versions.

---

## 📄 Documentation

A full **Software Requirements Specification (SRS)** document in IEEE format is available in this repository covering:
- System architecture and data flow
- Functional requirements (REQ-1 through REQ-29)
- Non-functional requirements (performance, security, quality)
- Glossary of all ML and quantum computing terms used

---

## 📚 References

- [KDD Cup 1999 Dataset — UCI ML Repository](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/)

---

## 📝 License

This project is developed for academic and research purposes. Dataset usage is subject to the [KDD Cup 1999 terms](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).  
SRS template adapted from Karl E. Wiegers (1999) — free to use, modify, and distribute.

---

## 🙋 Author

**Nilanjana Jamindar**  
Department of Computer Science and Engineering  
RV University — 2026
