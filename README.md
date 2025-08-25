# fraud-detection-capstone

This is our final-year capstone project focused on detecting fraudulent credit card transactions using advanced machine learning techniques. We tackle the **class imbalance problem**, and compare various **models** to find the most effective fraud detection strategy.

---

## 👥 Team Members

- Zuleykha Salahova
- Abdurrahman Begovic 
- Marah Hasan
  
---

## 📁 Project Structure

```
fraud-detection-capstone/
│
├── demo/                # Demo API / prototype code (FastAPI, etc.)
├── notebooks/           # Jupyter notebooks for model training and experiments
├── results/             # Saved model results, metrics, and figures
├── scripts/             # Python scripts for preprocessing, training, and evaluation
│
├── .gitignore          
├── README.md            # Project documentation
├── package-lock.json    
├── package.json         
├── requirements.txt     # Python dependencies
```

---

## 📊 Dataset

- **Name:** Synthetic Credit Card Fraud Detection Dataset
- **Source:** https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
- **File used:** `card_transdata.csv`
- Contains anonymized transaction data with class imbalance between normal and fraudulent transactions.

---

## 🚀 Demo

To run the demo, follow these steps:
1. **Dataset:** Ensure the above dataset is stored at ./data/card_transdata.csv
2. **Preprocessing:** Run the preprocessing script using:
```bash
python ./scripts/preprocessing.py
```
3. **Model Training:** Within ./notebooks/, run all the cells within each Jupyter Notebook, except for preprocessing.ipynb
4. **Demo Prep:** To prepare all files for the demo, run:
```bash
python ./scripts/prep_demo_files.py
```
5. **Run:** Using the terminal, enter the demo folder and run the API:
```bash
cd demo
uvicorn main:app --host 0.0.0.0 --port 8080
```
6. **Access:** The demo page can be access at http://localhost:8080/home

---

## 📬 Contact

If you have any questions or feedback, feel free to contact us through email.

---

⭐️ Thank you for checking out our project!
