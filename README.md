
# Formula 1 DNF Prediction  
Western Governors University – C964 Computer Science Capstone  

This project uses machine learning to predict whether a Formula 1 driver will record a **DNF (Did Not Finish)** in a race.  
By analyzing historical race data, including driver, constructor, weather, and circuit information, the model helps identify reliability risks before race day.

## Author

**Nicholas Halterman**  
Western Governors University – B.S. Computer Science  
Capstone Project (C964)

## Project Overview

**Goal:**  
Build and evaluate a supervised machine learning model that classifies race outcomes as **Finish (0)** or **DNF (1)**.

**Dataset:**  
[F1 DNF Classification – Kaggle](https://www.kaggle.com/datasets/pranay13257/f1-dnf-classification)

**Tech Stack:**  
Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib, Jupyter Notebook

**Models Used:**  
Logistic Regression, Random Forest, XGBoost

**Evaluation Metrics:**  
Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

## Installation

**Please note that for the install to work properly, please use Python 3.11.**

1. Clone the repository:
   ```bash
   git clone https://github.com/nickhalterman/f1_dnf_predictor.git
   cd f1-dnf-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Running the Notebook

1. Open `f1_dnf_predictor.ipynb`.
2. Run all cells in order.  
   The notebook will:
   - Load and clean the dataset  
   - Train multiple models  
   - Evaluate their performance  
   - Save the best model to `/src/model.joblib`

## Model Prediction and Interactive Testing

After training and saving the model, you can test predictions interactively inside the notebook.  
The cell labeled **"Interactive DNF prediction with ipywidgets"** allows you to explore results directly from the test dataset.

### How to Use

1. Ensure the model and `X_test` dataset are loaded:
   ```python
   from joblib import load
   model = load("src/model.joblib")
   ```

2. Scroll to the interactive prediction cell and run it.  
   A slider labeled **"Sample Index"** will appear.

3. Use the slider or type a number in the input box to select a record from `X_test`.

4. Each time you change the index, the notebook will display:
   - The selected record index  
   - The model’s prediction (**DNF** or **Finish**)  
   - The probability of each outcome  
   - The feature values for that record

In this example, the model predicts that the driver will not finish the race with an estimated 85.6% probability of DNF.  
You can continue adjusting the index to explore how the model performs across different samples.

## Project Files

| File | Description |
|------|--------------|
| `f1_dnf_predictor.ipynb` | Main Jupyter Notebook containing all code and analysis |
| `src/model.joblib` | Saved trained model |
| `data/f1_dnf.csv` | Dataset used for training and testing |
| `requirements.txt` | List of dependencies |

## Notes

- Run the entire notebook from top to bottom for consistent results.  
- The dataset file must be located in the `data/` folder.  
- You can adjust model hyperparameters to improve performance.

