# Fraud-Detection-for-Online-Bank-Transaction-Using-GAN
# 💻 Fraud Detection for Online Bank Transactions using GAN Framework

## Project Summary
This project uses a Generative Adversarial Network (GAN) to generate synthetic fraudulent transactions to address class imbalance, then trains classical ML models (SVM, KNN, Logistic Regression) on the augmented dataset for fraud detection.

**Key results (from tests):** Precision 92%, Recall 88%, F1 90%, ROC-AUC 0.94. (See Project_Report.pdf for full tables). :contentReference[oaicite:4]{index=4}

---

## Repository structure
- `src/` — preprocessing, GAN training, ML training, evaluation, API serve scripts  
- `models/` — saved model files (GAN generator, classifier .pkl / .onnx)  
- `Dataset/` — sample or anonymized dataset (do not commit PII)  
- `notebooks/` — EDA and experiments  
- `docs/` — diagrams and screenshots  
- `Project_Report.pdf` — full project report.

---

## How to run (local demo)
```bash
# 1. prepare environment
python -m venv venv
# linux/mac
source venv/bin/activate
# windows
venv\Scripts\activate

pip install -r requirements.txt

# 2. preprocess (example)
python src/preprocess.py --input Dataset/sample_dataset.csv --output Dataset/processed.csv

# 3. train GAN (if needed)
python src/train_gan.py --data Dataset/processed.csv --out models/gan_generator.pth

# 4. train classifiers
python src/train_models.py --data Dataset/processed_augmented.csv --out models/

# 5. evaluate
python src/evaluate.py --models models/ --test Dataset/test.csv

# 6. serve API
python src/serve.py
# then open http://localhost:5000/predict to demo
