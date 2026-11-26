# 📱 Mobile Price Prediction System

## Introduction

This project is a **Mobile Price Prediction System** developed using **Machine Learning and Data Analysis techniques**.

The main objective of this project is to predict the price category of a mobile phone based on its technical specifications and hardware features.

Mobile phone prices depend on various factors such as RAM, battery capacity, processor, camera quality, internal memory, and screen specifications.

This project uses Machine Learning classification algorithms to analyze mobile phone specifications and classify devices into four different price categories:

- **0 – Low**
- **1 – Medium**
- **2 – High**
- **3 – Very High**

The system helps understand how different mobile specifications influence the overall price category.

---

## Methods

### 1. Data Collection

A mobile phone specifications dataset is used to develop the prediction model.

The dataset contains information about mobile phone hardware specifications and their corresponding price categories.

Important features include:

- Battery Power
- RAM
- Internal Memory
- Processor Cores
- Processor Speed
- Front Camera
- Primary Camera
- Screen Width
- Screen Height
- Pixel Resolution

### 2. Data Preprocessing

Data preprocessing is performed to prepare the dataset for Machine Learning.

The preprocessing steps include:

- Handling missing values
- Removing duplicate records
- Identifying and handling outliers
- Selecting relevant features
- Converting data into suitable numerical formats
- Splitting data into training and testing datasets
- Feature scaling using StandardScaler

These steps help improve data quality and model performance.

### 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the relationship between mobile specifications and their price categories.

The analysis includes:

- Price category distribution
- RAM and price relationship
- Battery power analysis
- Internal memory analysis
- Feature correlation
- Feature importance

Different visualization techniques, including correlation heatmaps and boxplots, are used to understand the dataset.

### 4. Feature Engineering

Feature engineering is performed to create additional useful features from existing mobile specifications.

The derived features include:

- Screen Size
- Pixel Density
- Average Camera Resolution
- Memory per Core
- Power Efficiency

These features help the model understand relationships between mobile specifications and price categories.

### 5. Model Training

Multiple Machine Learning classification algorithms are implemented and compared.

The algorithms used include:

- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

The dataset is divided into training and testing sets using stratified sampling.

Cross-validation and GridSearchCV are used to evaluate model performance and optimize hyperparameters.

### 6. Model Evaluation

The trained models are evaluated using different classification metrics.

The evaluation techniques include:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross-Validation Score
- Classification Report

These metrics help identify the best-performing model for mobile price classification.

---

## Output

The trained Machine Learning model predicts the price category of a mobile phone based on its hardware specifications.

### Prediction Workflow

```text
Mobile Phone Dataset
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Feature Scaling
        ↓
Train-Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Mobile Price Prediction
```

### Price Categories

The system classifies mobile phones into four categories:

| Class | Price Category |
|---|---|
| 0 | Low |
| 1 | Medium |
| 2 | High |
| 3 | Very High |

### Model Results

Multiple classification algorithms are compared to identify the best-performing model.

The evaluation results include accuracy, confusion matrix, precision, recall, and F1-score.

Random Forest and Gradient Boosting demonstrate strong classification performance in the project.

### Application Output

The project also includes a Streamlit-based application that allows users to provide mobile phone specifications and obtain the predicted price category.

---

## Conclusion

This project demonstrates how **Machine Learning classification algorithms** can be used to predict mobile phone price categories based on technical specifications.

By applying data preprocessing, exploratory data analysis, feature engineering, and multiple classification algorithms, the system identifies patterns between mobile hardware features and their corresponding price categories.

The project also demonstrates model comparison, hyperparameter optimization, and application deployment using Streamlit.

The system can be further improved by:

- Using larger and more diverse mobile phone datasets
- Exploring advanced ensemble learning algorithms
- Improving feature engineering
- Performing additional hyperparameter optimization
- Incorporating real-time mobile market data
- Enhancing the Streamlit application

Overall, this project provides a practical implementation of **Machine Learning for Mobile Price Classification**.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- GridSearchCV
- Streamlit

---
