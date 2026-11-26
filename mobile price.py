import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

# ============================================
# 1. DATA LOADING
# ============================================
def load_data(filepath):
    """
    Load the mobile phone dataset from CSV file
    Update the filepath to your actual dataset location
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at {filepath}")
        print("Please update the filepath to your dataset location")
        return None

# Update this path to your actual file location
# filepath = r"C:\Users\VIAMRSH P S\Desktop\Projects-20240722T093004Z-001\Projects\mobile_phone_pricing\Mobile Phone Pricing\dataset.csv"
# df = load_data(filepath)

# For demonstration, if file doesn't load, create sample structure
# df = load_data("dataset.csv")

# ============================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================
def perform_eda(df):
    """Comprehensive EDA of the mobile phone dataset"""
    
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Basic info
    print("\n1. Dataset Info:")
    print(df.info())
    
    print("\n2. First 5 rows:")
    print(df.head())
    
    print("\n3. Statistical Summary:")
    print(df.describe())
    
    print("\n4. Missing Values:")
    print(df.isnull().sum())
    
    print("\n5. Duplicate Rows:", df.duplicated().sum())
    
    print("\n6. Target Variable Distribution:")
    print(df['price_range'].value_counts().sort_index())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Target distribution
    ax1 = axes[0, 0]
    df['price_range'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Distribution of Price Ranges')
    ax1.set_xlabel('Price Range (0=Low, 1=Medium, 2=High, 3=Very High)')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    # Correlation heatmap
    ax2 = axes[0, 1]
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Feature Correlation Heatmap')
    
    # RAM vs Price Range (Most important feature)
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='price_range', y='ram', ax=ax3)
    ax3.set_title('RAM Distribution by Price Range')
    ax3.set_xlabel('Price Range')
    ax3.set_ylabel('RAM (MB)')
    
    # Battery Power vs Price Range
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x='price_range', y='battery_power', ax=ax4)
    ax4.set_title('Battery Power Distribution by Price Range')
    ax4.set_xlabel('Price Range')
    ax4.set_ylabel('Battery Power (mAh)')
    
    plt.tight_layout()
    plt.show()
    
    # Feature correlation with target
    print("\n7. Correlation with Target (Price Range):")
    target_corr = df.corr()['price_range'].drop('price_range').sort_values(ascending=False)
    print(target_corr)
    
    return target_corr

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
def engineer_features(df):
    """Create new features and prepare data for modeling"""
    
    df = df.copy()
    
    # Create screen size feature (area)
    df['screen_size'] = df['sc_h'] * df['sc_w']
    
    # Create pixel density feature (pixels per cm)
    df['pixel_density'] = (df['px_height'] * df['px_width']) / df['screen_size']
    df['pixel_density'] = df['pixel_density'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Create camera quality feature (average of front and primary)
    df['avg_camera'] = (df['fc'] + df['pc']) / 2
    
    # Create memory per core feature
    df['memory_per_core'] = df['int_memory'] / df['n_cores']
    
    # Create power efficiency feature
    df['power_efficiency'] = df['battery_power'] / df['talk_time']
    df['power_efficiency'] = df['power_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    print("Feature engineering completed!")
    print(f"New features added: screen_size, pixel_density, avg_camera, memory_per_core, power_efficiency")
    print(f"New dataset shape: {df.shape}")
    
    return df

# ============================================
# 4. DATA PREPROCESSING
# ============================================
def preprocess_data(df, feature_cols, target_col='price_range'):
    """Split and scale the data"""
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Scale features (important for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# ============================================
# 5. MODEL BUILDING & TRAINING
# ============================================
def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train multiple classification models and compare performance"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\n{'-'*40}")
        print(f"Training {name}...")
        
        # Use scaled data for SVM and Logistic Regression, original for tree-based
        if name in ['Support Vector Machine', 'Logistic Regression', 'K-Nearest Neighbors']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            # Cross-validation on scaled data
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Cross-validation on original data
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    
    return results

# ============================================
# 6. MODEL EVALUATION & COMPARISON
# ============================================
def evaluate_models(results, y_test):
    """Detailed evaluation and visualization of model performance"""
    
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'CV Mean': [results[m]['cv_mean'] for m in results.keys()],
        'CV Std': [results[m]['cv_std'] for m in results.keys()]
    }).sort_values('Test Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    ax1 = axes[0]
    comparison_df.plot(x='Model', y=['Test Accuracy', 'CV Mean'], kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Best model confusion matrix
    best_model_name = comparison_df.iloc[0]['Model']
    best_predictions = results[best_model_name]['predictions']
    
    ax2 = axes[1]
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Confusion Matrix - {best_model_name}\n(Best Model)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed classification report for best model
    print(f"\n{'='*50}")
    print(f"DETAILED REPORT FOR BEST MODEL: {best_model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, best_predictions, 
                               target_names=['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost']))
    
    return best_model_name, comparison_df

# ============================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================
def analyze_feature_importance(best_model, feature_cols, X_train, y_train):
    """Analyze which features are most important for prediction"""
    
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Get feature importances if available (tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature', palette='viridis')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
    else:
        # For SVM and other models without built-in feature importance
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(best_model, X_train, y_train, n_repeats=10, random_state=RANDOM_STATE)
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': result.importances_mean
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features (Permutation Importance):")
        print(feature_importance_df.head(10).to_string(index=False))
    
    return feature_importance_df

# ============================================
# 8. HYPERPARAMETER TUNING
# ============================================
def tune_best_model(best_model_name, X_train, y_train, X_train_scaled, y_train_scaled=None):
    """Perform hyperparameter tuning on the best model"""
    
    print("\n" + "=" * 50)
    print(f"HYPERPARAMETER TUNING: {best_model_name}")
    print("=" * 50)
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        
    elif best_model_name == 'Support Vector Machine':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        model = SVC(random_state=RANDOM_STATE)
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        search.fit(X_train_scaled, y_train)
        
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
    
    else:
        print("Hyperparameter tuning not implemented for this model")
        return None
    
    print(f"\nBest Parameters: {search.best_params_}")
    print(f"Best CV Score: {search.best_score_:.4f}")
    
    return search.best_estimator_

# ============================================
# 9. PREDICTION FUNCTION
# ============================================
class MobilePricePredictor:
    """Class to handle new predictions"""
    
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.price_mapping = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
        
    def predict_single(self, phone_features):
        """
        Predict price range for a single phone
        
        phone_features: dict with keys matching feature_cols
        """
        # Convert to dataframe
        input_df = pd.DataFrame([phone_features])
        
        # Ensure correct column order
        input_df = input_df[self.feature_cols]
        
        # Scale if needed
        if isinstance(self.model, (SVC, LogisticRegression, KNeighborsClassifier)):
            input_scaled = self.scaler.transform(input_df)
            prediction = self.model.predict(input_scaled)[0]
        else:
            prediction = self.model.predict(input_df)[0]
        
        return {
            'price_class': int(prediction),
            'price_range': self.price_mapping[prediction],
            'confidence': 'High' if hasattr(self.model, 'predict_proba') else 'N/A'
        }
    
    def predict_batch(self, phones_df):
        """Predict price ranges for multiple phones"""
        predictions = []
        
        for _, row in phones_df.iterrows():
            pred = self.predict_single(row.to_dict())
            predictions.append(pred)
        
        return pd.DataFrame(predictions)

# ============================================
# MAIN EXECUTION PIPELINE
# ============================================
def main():
    """Complete pipeline for mobile price prediction"""
    
    # Update this path to your actual dataset location
    filepath = r"C:\Users\VIAMRSH P S\Desktop\Projects-20240722T093004Z-001\Projects\mobile_phone_pricing\Mobile Phone Pricing\dataset.csv"
    
    # 1. Load Data
    print("Step 1: Loading Data...")
    df = load_data(filepath)
    
    if df is None:
        print("Please update the filepath variable with your correct dataset path")
        return
    
    # 2. EDA
    target_corr = perform_eda(df)
    
    # 3. Feature Engineering
    print("\nStep 2: Feature Engineering...")
    df_engineered = engineer_features(df)
    
    # 4. Prepare Features
    # Select all columns except target
    feature_cols = [col for col in df_engineered.columns if col != 'price_range']
    
    # Or select top correlated features (optional)
    top_features = target_corr.head(10).index.tolist()
    feature_cols = [f for f in top_features if f in df_engineered.columns]
    
    print(f"\nUsing features: {feature_cols}")
    
    # 5. Preprocess
    print("\nStep 3: Preprocessing...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_data(
        df_engineered, feature_cols
    )
    
    # 6. Train Models
    print("\nStep 4: Training Models...")
    results = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # 7. Evaluate
    print("\nStep 5: Evaluating Models...")
    best_model_name, comparison_df = evaluate_models(results, y_test)
    
    # 8. Feature Importance
    best_model = results[best_model_name]['model']
    feature_importance_df = analyze_feature_importance(best_model, feature_cols, X_train, y_train)
    
    # 9. Hyperparameter Tuning (Optional - takes time)
    print("\nStep 6: Tuning Best Model...")
    tuned_model = tune_best_model(best_model_name, X_train, y_train, X_train_scaled, y_train)
    
    # 10. Create Predictor Instance
    print("\nStep 6: Creating Prediction System...")
    predictor = MobilePricePredictor(best_model, scaler, feature_cols)
    
    # Example prediction
    print("\nExample Prediction:")
    sample_phone = df_engineered.iloc[0][feature_cols].to_dict()
    sample_phone.pop('price_range', None)  # Remove target if present
    prediction = predictor.predict_single(sample_phone)
    print(f"Input Features: {sample_phone}")
    print(f"Predicted Price Range: {prediction}")
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.2%}")
    print("\nYou can now use the 'predictor' object to make new predictions!")
    
    return predictor, results, feature_importance_df

# Run the pipeline
if __name__ == "__main__":
    predictor, results, feature_importance = main()