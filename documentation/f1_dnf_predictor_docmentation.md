# Formula 1 DNF Predictor Documentation

**C964 - Computer Science Capstone**

**Western Governors University (WGU)**

Nicholas Halterman (011576257)

# Part A - Letter of Transmittal

**October 23, 2025**

Formula 1 Race Strategy Team

Fédération Internationale de l’Automobile (FIA)

Place de la Concorde

75008 Paris, France

Dear Race Strategy Team,

Formula 1 is the pinnacle of motorsport, a blend of engineering excellence, data precision, and strategic mastery. Yet, despite the technological sophistication of modern teams, one area remains notoriously unpredictable which is the driver recording a *Did Not Finish (DNF)* in a race. Mechanical failures, collisions, and environmental factors can abruptly alter the outcome of a Grand Prix, costing teams valuable championship points and millions in resources.

To address this, I developed the Formula 1 DNF Predictor, a machine learning application that estimates the probability a driver will fail to finish a race. The model analyzes historical Formula 1 race data, including factors such as the circuit, constructor, grid position, season, to deliver pre-race reliability insights that help teams make better strategic and risk management decisions.

This project will deliver a Jupyter-based analytical tool capable of generating DNF probability predictions using historical Formula 1 datasets. It incorporates a trained Logistic Regression model, a data visualization dashboard, and a user-friendly interface for testing predictions from the trained model. The anticipated cost to develop and maintain this system is approximately $24,500 over six months, covering development time, cloud compute resources, and dataset storage.

The benefits to the organization are substantial. Strategists and engineers will gain access to a predictive tool that highlights potential risk scenarios before they happen. A more data-driven reliability analysis reduces unexpected retirements, optimizes component allocation, and improves the team’s overall competitive standing. Furthermore, the project is ethically sound, utilizing open-source, publicly available racing datasets with no personal or proprietary data.

Thank you for your consideration of this proposal. I am confident the Formula 1 DNF Predictor will become a valuable addition to your strategic toolkit, combining machine learning with motorsport expertise to improve decision-making and reduce uncertainty.

Sincerely,

Nicholas Halterman

# Part B - Project Proposal Plan

### Project Summary

The Formula 1 Race Strategy Team currently faces a significant challenge which is predicting whether a driver will complete or fail to finish a race. DNFs can arise from mechanical issues, driver errors, or environmental conditions, often resulting in unpredictable outcomes. These incidents directly affect constructor standings and financial performance.

The proposed solution is the Formula 1 DNF Predictor, a supervised machine learning application that analyzes historical Formula 1 race data to forecast the likelihood of DNFs. The project’s deliverables include a trained predictive model, three analytical visualizations (feature importance, confusion matrix, ROC curve), and a user guide for implementation. This predictive capability will provide insights for the strategy teams to quantify risk and prepare contingency plans before each race weekend.

By integrating this tool into pre-race strategy briefings, the organization can make more informed decisions regarding driver set up, tire strategy, and mechanical reliability, which will contribute to better race outcomes and reduced retirements.

### Data Summary

The data for this project originates from the “F1 DNF Classification” dataset hosted on Kaggle, which includes race-by-race results spanning multiple seasons. The dataset provides information such as driver names, constructors, grid positions, circuits, and finishing status (0 = Finished, 1 = DNF).

Data was cleaned and preprocessed in Python using pandas and scikit-learn. Missing or incomplete fields were imputed or removed to preserve dataset integrity. Features were encoded numerically to support machine learning models. Outliers, such as races with no recorded results, were excluded from training data to prevent bias.

The dataset meets all ethical and legal standards. No personal or proprietary data is used. All data is public and intended for research and educational purposes.

### Implementation

The Waterfall Method will be used, given the project’s clearly defined requirements and linear workflow. The project will proceed through five structured phases:

1. **Requirements Analysis:** Identify relevant race metrics and confirm data scope.
2. **Design:** Develop preprocessing pipelines and define model evaluation metrics (accuracy, F1-score, ROC)
3. **Implementation:** Train Logistic Regression model using scikit-learn.
4. **Verification:** Test the model on unseen data to validate performance.
5. **Maintenance:** Update datasets annually and retrain models as new race data become available.

The model will be developed in Jupyter Notebook, and results will be exported using joblib for model persistence.

### Timeline

| Milestone | Dependencies | Resources | Start-End | Duration |
| --- | --- | --- | --- | --- |
| Requirements Gathering | None | Developer | Nov 1 - Nov 7 | 1 week |
| Data Cleaning & EDA | Dataset | Developer | Nov 8 - Nov 21 | 2 weeks |
| Model Training | Preprocessed Data | Developer | Nov 22 - Dec 15 | 3 weeks |
| Model Evaluation | Trained Model | Developer | Dec 16 - Dec 23 | 1 week |
| Visualization & UI | Model Output | Developer | Jan 2 - Jan 16 | 2 weeks |
| Documentation & Testing | Complete Model | Developer  | Jan 17 - Jan 23 | 1 week |
| Deployment & Handoff | Final Build | Developer | Jan 24 - Feb 6 | 2 weeks |

### Evaluation Plan

Verification will occur at each stage:

- **Data Verification:** Confirm no missing or mislabled data records.
- **Code Verification:** Peer review for reproducibility and logic accuracy.
- **Model Verification:** Test accuracy, precision, recall, F1-score, and ROC on test data.
- **User Validation:** Ensure the Jupyter interface runs correctly and predictions match expected logic.

Final validation will rely on achieving at least 75% accuracy and an F1-score above 0.8, ensuring balance performance between predicting DNFs and finishes.

### Resources and Costs

| Resource | Description | Cost |
| --- | --- | --- |
| Developer Labor | ~240 hours @ $100/hour | $24,000 |
| Development Environment | Jupyter Notebook, Python, Pandas, Scikit-learn | $0 |
| Cloud Compute | Model training on cloud GPU instances | $300 |
| Dataset | Kaggle “F1 DNF Classification” dataset | $0 |
| Maintenance | Yearly retraining and updates | $200/year |

**Total Initial Cost: $24,500**

# Part C - Application

The Formula 1 DNF Predictor application consists of a trained machine learning model, a Jupyter Notebook interface, and visualizations to analyze performance. The model was developed in Python using pandas, numpy, and scikit-learn, and then stored using joblib for later use.

**Submitted files include:**

- f1_dnf_predictor.ipynb - Full notebook with EDA, training, and testing
- model.joblib - Saved Logistic Regression model
- src/visualizations/ - Includes season histogram,  confusion matrix, ROC curve, and feature important plots
- README - Setup and usage instructions
- requirements.txt - Dependency list

The model accepts driver and race data as input and outputs a binary classification (0 = Finish, 1 = DNF) along with a probability score.

# Part D - Post Implementation Report

### Solution Summary

Before this project, no predictive tool existed to help the Formula 1 Race Strategy Team anticipate DNFs, based on data-driven analysis. The developed application solves this problem by providing a model that analyzes historical race data to forecast DNF likelihood for each driver. The tool helps strategist evaluate reliability risk and make decisions prior to race day.

### Data Summary

The dataset used came from Kaggle’s “F1 DNF Classification”, which includes historical race results and additional data. After importing the dataset into Python, columns with missing or incomplete values were handled appropriately such as dropping non-critical rows and encoding categorical variables.

The data was then split into training (80%) and testing (20%) subsets. It was stored in CSV format and access locally from the Jupyter environment for reproducibility.

### Machine Learning

The application used a **Logistic Regression classifier**, chosen for its simplicity, interpretability, and effectiveness in binary classification problems such as predicting whether a driver will finish or record a DNF.

The Logistic Regression model achieved an accuracy of 75%, an F1-score of 0.823, and an ROC-AUC of 0.849.

The model was trained using key features such as driver, constructor, circuit, starting grid position, and race year. The Logistic Regression model was used to capture relationships between these variables and the likelihood of a DNF. Feature importance can be examined to understand which race and constructor factors most influence the likelihood of DNFs.

### Validation

The Formula 1 DNF Predictor was a supervised-machine-learning project designed to classify race outcomes either as a Finish (0) or Did Not Finish (1).

Validation of the model performance was completed using a combination of metrics and visual evaluation.

**Validation Method:**

The training data was used to fit the Logistic Regression Classifier, while the held-out testing data was reserved exclusively for validation. This ensured that evaluation results reflected the model’s ability to generalize to unseen races. Cross-validation was also performed to confirm consistency of accuracy across random splits.

**Performance Metrics:**

Standard classification metrics were computed with scikit-learn’s built-in evaluation features. 

- **Accuracy:**  Proportion of total predictions correctly classified.
- **Precision:** Percentage of positive DNF predictions that were correct.
- **Recall:** Percentage of actual DNFs that were correctly identified.
- **F-1 Score:** Harmonic mean of precision and recall, balancing both types of error.
- **ROC:** Area under the Receiver Operating Characteristic curve, measuring discrimination between Finish and DNF.

**Results:**

The final Logistic Regression model achieved an accuracy of 0.76, F1-score of 0.823, and ROC of 0.849 on the test data. These values confirmed that the model successfully captured relationships among driver, constructor, circuit, grid position, and season variables while maintaining low misclassification error. Visual review of the confusion matrix verified that the model correctly classified most race outcomes, with only a small number of false positives and false negatives.

**Interpretation**

The metrics demonstrate strong generalization in unseen races. The relatively high F1-score indicates a balanced trade-off between identifying real DNFs and avoiding incorrect DNF predictions. The ROC near 0.85 shows that the classifier reliability distinguishes finishing and non-finishing competitors, validating the model as suitable for strategic risk analysis.

### Visualizations

Several visualizations were created to better understand both the dataset and the model’s predictive performance.

A histogram of race entries per season confirmed that data was evenly distributed across years, ensuring no season dominated the training set. A correlation heatmap helped verify that no predictors were overly correlated, which reduces the risk of redundant features influencing the model.

For model evaluation, a confusion matrix was generated to visualize the number of correctly and incorrectly classified race outcomes. The matrix showed that most predictions fell along the diagonal, indicating strong agreement between predicted and actual outcomes. Finally, a ROC curve was plotted to visualize the trade-off between true positive and false positive rates. The area under the curve of 0.849 demonstrated that the Logistic Regression model performed reliably and maintained good judgement between finishers and non-finishers.

### User Guide

The Formula 1 DNF Predictor can be executed locally through Jupyter Notebook or accessed via the project’s GitHub repository. The following steps describe how to install dependencies, run the notebook, and test model predictions.

1. Clone the repository
    
    Open a terminal or command prompt and run the following:
    
    ```bash
    git clone https://github.com/nickhalterman/f1_dnf_predictor.git
    cd f1_dnf_predictor
    ```
    
2. Set up a Virtual Environment (Please use Python 3.11)
    
    Create and activate a Python virtual environment:
    
    **Windows**
    
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    
    **macOS/Linux**
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    
3. Install Dependencies
    
    Install all required packages:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    If Jupyter Notebook is not included, install it manually:
    
    ```bash
    pip install jupyter
    ```
    
4. Launch Jupyter Notebook
    
    Start Jupyter from the terminal:
    
    ```bash
    jupyter notebook
    ```
    
    Open the file **f1_dnf_predictor.ipynb** **from the project directory.
    
5. Run the Notebook
    1. Execute all cells sequentially from top to bottom.
    2. The notebook will
        - Automatically load and preprocess the dataset
        - Train the Logistic Regression model
        - Evaluate the model’s accuracy, precision, recall, F1-score, and ROC
        - Save the trained model to src/model.joblib for later use
    
    After training completes, you’ll see printed metrics and visualizations that summarize the model’s performance.
    
6. Test a prediction
    1. Scroll to the “Using the Interactive DNF Predictor” cell.
    2. Run the cell to load the trained model.
    3. Use the slider or enter an index value to select a sample from the test dataset.
    4. The notebook will display
        1. The driver’s race record (from the selected test sample)
        2. The model’s predicted outcome (Finish or DNF)
        3. The probability of each outcome
    
    This allows users to explore how the model responds to different race conditions and better understand what factors may influence a driver’s likelihood of finishing a race.
    

### References

- Pranay13257. (2022). *F1 DNF Classification* [Dataset]. Kaggle. https://www.kaggle.com/datasets/pranay13257/f1-dnf-classification
