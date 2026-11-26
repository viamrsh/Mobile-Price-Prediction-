# Mobile-Price-Prediction-
Build a system that can predict pricing for a mobile phone using data on available phones in the market. Predict if the mobile can be priced low/med/high/very high. Explore the data to understand the features and figure out an approach.
# Introduction 
In today’s competitive smartphone market, pricing plays a crucial role in product positioning and 
customer decision-making. Mobile phone prices depend on multiple hardware and performance-related 
features such as RAM, battery capacity, camera quality, internal memory, processor cores, and display 
specifications. The objective of this project is to develop a machine learning-based prediction system 
that classifies mobile phones into different price ranges (Low, Medium, High, Very High) based on their 
technical specifications. By leveraging supervised learning algorithms, the system aims to analyze 
patterns in historical data and accurately predict the price category of new mobile devices. This project 
demonstrates the application of data preprocessing, feature engineering, model training, and 
performance evaluation to build a reliable predictive model. 
# Methodology 
The methodology follows a structured machine learning pipeline: 
Data Collection and Exploration: 
The dataset containing mobile specifications and corresponding price ranges is loaded and analyzed. 
Exploratory Data Analysis (EDA) is performed to understand feature distributions, check missing 
values, detect duplicates, and analyze correlations between features and the target variable. 
Visualization techniques such as correlation matrices and boxplots help identify important features like 
RAM and battery power. 
Feature Engineering: 
New features are created to improve model performance. Examples include screen size (screen height 
× width), pixel density, average camera resolution, memory per core, and power efficiency. These 
derived features help capture complex relationships between hardware specifications and price range. 
Data Preprocessing: 
The dataset is split into training and testing sets using stratified sampling to maintain class balance. 
Feature scaling is applied using StandardScaler for algorithms sensitive to feature magnitude. 
Model Training: 
Multiple classification algorithms are implemented, including: 
• Random Forest 
• Gradient Boosting 
• Support Vector Machine (SVM) 
• Logistic Regression 
• Decision Tree 
• K-Nearest Neighbors (KNN) 
Cross-validation is used to evaluate model stability and prevent overfitting. 
Model Evaluation and Tuning: 
Performance is measured using accuracy, cross-validation mean score, confusion matrix, and 
classification report (precision, recall, F1-score). Hyperparameter tuning is performed using 
GridSearchCV to optimize selected models. 
# Output Results 
The trained models were evaluated on the test dataset. Among all classifiers, ensemble methods such as 
Random Forest and Gradient Boosting showed the highest accuracy and stability across cross-validation 
folds. The confusion matrix indicates strong classification performance across all price categories, with 
minimal misclassification between adjacent price ranges. 
Feature importance analysis revealed that RAM, battery power, pixel resolution, and internal memory 
are the most influential features in determining mobile price range. The final optimized model provides 
accurate predictions for both single and batch inputs using a custom-built prediction class. 
Overall, the system achieves high classification accuracy, demonstrating the effectiveness of ensemble 
learning methods for structured tabular datasets. 
# Conclusion 
This project successfully implements a complete machine learning pipeline for mobile price range 
prediction. By combining data exploration, feature engineering, preprocessing, and multiple 
classification models, a robust and accurate predictive system was developed. Ensemble models such 
as Random Forest and Gradient Boosting delivered superior performance due to their ability to handle 
non-linear relationships and feature interactions. 
The developed system can be extended further by incorporating larger datasets, advanced 
hyperparameter optimization techniques, and deployment as a web-based prediction service. This study 
highlights how machine learning techniques can effectively assist in market analysis, pricing strategies, 
and decision support systems in the mobile industry.
