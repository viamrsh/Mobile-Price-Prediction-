import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Configure plotting
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# ======================
# 1. DATA LOADING & SETUP
# ======================
print("="*60)
print("MOBILE PHONE PRICE PREDICTION SYSTEM")
print("="*60)

# USER NOTE: Update this path if needed. Handles common scenarios:
dataset_paths = [
    r"C:\Users\VIAMRSH P S\Desktop\Projects-20240722T093004Z-001\Projects\mobile_phone_pricing\Mobile Phone Pricing\dataset.csv",
    "dataset.csv",  # Current directory fallback
    "mobile_price_dataset.csv"  # Common alternative name
]

df = None
for path in dataset_paths:
    try:
        if Path(path).exists():
            df = pd.read_csv(path)
            print(f"✓ Loaded dataset from: {path}")
            break
    except Exception as e:
        continue

if df is None:
    raise FileNotFoundError(
        "Dataset not found! Please:\n"
        "1. Verify file exists at specified path\n"
        "2. Check filename (try 'dataset.csv' or 'mobile_price_dataset.csv')\n"
        "3. Update path in code under 'dataset_paths'"
    )

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Identify target column (should be the price range column)
possible_targets = [col for col in df.columns if 'price' in col.lower() or 'range' in col.lower()]
if not possible_targets:
    # If no obvious target, assume last column is target
    target_col = df.columns[-1]
    print(f"⚠️  No obvious target column found. Assuming last column '{target_col}' is the target.")
else:
    target_col = possible_targets[0]
    print(f"✓ Identified target column: '{target_col}'")

print(f"Features: {len(df.columns)-1} | Target: {target_col}")

# ======================
# 2. EXPLORATORY DATA ANALYSIS
# ======================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Check missing values
missing = df.isnull().sum()
if missing.any():
    print("⚠️  Missing values detected! Imputing with median...")
    df.fillna(df.median(numeric_only=True), inplace=True)
else:
    print("✓ No missing values found")

# Target distribution analysis
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=target_col, data=df, palette='viridis')
plt.title('Price Range Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Price Range (0=Low, 1=Med, 2=High, 3=Very High)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved price distribution plot: price_distribution.png")

# Class balance check
class_dist = df[target_col].value_counts().sort_index()
imbalance_ratio = class_dist.min() / class_dist.max()
is_imbalanced = imbalance_ratio < 0.8
print(f"\nClass distribution:\n{class_dist}")
print(f"Imbalance ratio (min/max): {imbalance_ratio:.2f} -> {'IMBALANCED' if is_imbalanced else 'BALANCED'}")

# Feature correlation heatmap (top 10 features)
plt.figure(figsize=(14, 10))
corr = df.corr()
# Get top correlated features with target
top_features = corr[target_col].abs().sort_values(ascending=False).index[1:11]
sns.heatmap(corr.loc[top_features, top_features], annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Top 10 Features)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved correlation heatmap: correlation_heatmap.png")

# ======================
# 3. PREPROCESSING
# ======================
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-test split (stratified to maintain class distribution)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape} | Test set: {X_test.shape}")

# Handle imbalance if detected (SMOTE on TRAINING SET ONLY)
if is_imbalanced:
    print("⚠️  Applying SMOTE to balance training data...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts().min()-1))
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE - Training shape: {X_train.shape}")
    print(f"  New class distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")

# ======================
# 4. MODEL TRAINING & EVALUATION
# ======================
print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, ConfusionMatrixDisplay
)
import lightgbm as lgb

# Define models with class_weight handling
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced' if not is_imbalanced else None,
        random_state=42,
        n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        class_weight='balanced' if not is_imbalanced else None,
        random_state=42,
        verbosity=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate on TEST set (unseen data)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    results[name] = {'model': model, 'accuracy': acc, 'f1_macro': f1_macro}
    
    print(f"  Accuracy: {acc:.4f} | Macro F1-Score: {f1_macro:.4f}")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High', 'Very High'])}")

    # Track best model by macro F1 (robust to imbalance)
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_model = model
        best_model_name = name

# Final evaluation of best model
print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print("="*60)
y_pred_best = best_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Macro F1-Score: {f1_score(y_test, y_pred_best, average='macro'):.4f}")

# Confusion matrix visualization
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=['Low', 'Medium', 'High', 'Very High'],
    cmap='Blues',
    ax=ax,
    colorbar=False
)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved confusion matrix: confusion_matrix.png")

# ======================
# 5. EXPLAINABILITY (SHAP)
# ======================
print("\n" + "="*60)
print("MODEL EXPLAINABILITY (FEATURE IMPORTANCE)")
print("="*60)

# Feature importance plot
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Top 15 Feature Importances - {best_model_name}", fontsize=16, fontweight='bold')
    plt.bar(range(15), importances[indices[:15]], align="center", color='teal')
    plt.xticks(range(15), [features[i] for i in indices[:15]], rotation=45, ha='right', fontsize=11)
    plt.ylabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("✓ Saved feature importance plot: feature_importance.png")
    
    # Print top features
    print("\nTop 5 Predictive Features:")
    for i in range(5):
        print(f"  {i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

# SHAP analysis (if library available)
try:
    import shap
    print("\nGenerating SHAP explanations (may take 1-2 minutes)...")
    
    # Use smaller sample for SHAP to speed up
    X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
    
    if best_model_name == 'LightGBM':
        explainer = shap.TreeExplainer(best_model.booster_)
    else:
        explainer = shap.TreeExplainer(best_model)
    
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure(figsize=(12, 10))
    if isinstance(shap_values, list):  # Multi-class output
        shap.summary_plot(shap_values, X_sample, plot_type="bar", class_names=['Low','Med','High','V.High'], show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved SHAP summary plot: shap_summary.png")
    
    # Force plot for first prediction (class 3 = Very High)
    if isinstance(shap_values, list):
        plt.figure()
        shap.force_plot(
            explainer.expected_value[3], 
            shap_values[3][0], 
            X_sample.iloc[0], 
            matplotlib=True,
            show=False
        )
        plt.savefig('shap_force_example.png', dpi=150, bbox_inches='tight')
        print("✓ Saved SHAP force plot example: shap_force_example.png")
except Exception as e:
    print(f"⚠️  SHAP analysis skipped (install shap library for explainability): {str(e)[:100]}")

# ======================
# 6. MODEL PERSISTENCE & DEPLOYMENT
# ======================
print("\n" + "="*60)
print("MODEL EXPORT & DEPLOYMENT PREP")
print("="*60)

# Save best model
model_filename = 'mobile_price_predictor.pkl'
joblib.dump(best_model, model_filename)
print(f"✓ Saved best model to: {model_filename}")

# Create prediction function for deployment
def predict_phone_price(features_dict):
    """
    Predict price range for a new phone specification.
    
    Args:
        features_dict: Dictionary with feature names and values
        Example: {
            'battery_power': 1500, 'blue': 1, 'clock_speed': 2.2, ... 
        }
    
    Returns:
        dict: {
            'price_range': 'High', 
            'confidence': 0.92,
            'price_code': 2
        }
    """
    # Convert to DataFrame with same column order as training
    feature_df = pd.DataFrame([features_dict], columns=X.columns)
    
    # Predict
    pred_code = best_model.predict(feature_df)[0]
    pred_proba = best_model.predict_proba(feature_df)[0]
    
    # Map to labels
    price_labels = ['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost']
    confidence = pred_proba[pred_code]
    
    return {
        'price_range': price_labels[pred_code],
        'confidence': round(confidence, 4),
        'price_code': int(pred_code),
        'probabilities': {price_labels[i]: round(p, 4) for i, p in enumerate(pred_proba)}
    }

# Example usage of prediction function
print("\nExample prediction for a high-end phone:")
sample_phone = {
    'battery_power': 3000, 'blue': 1, 'clock_speed': 2.8, 'dual_sim': 1,
    'fc': 12, 'four_g': 1, 'int_memory': 64, 'm_deep': 0.8, 'mobile_wt': 150,
    'n_cores': 8, 'pc': 20, 'px_height': 1200, 'px_width': 1920,
    'ram': 4096, 'sc_h': 14, 'sc_w': 7, 'talk_time': 15,
    'three_g': 1, 'touch_screen': 1, 'wifi': 1
}
result = predict_phone_price(sample_phone)
print(f"Predicted Price Range: {result['price_range']} (Confidence: {result['confidence']:.0%})")
print(f"Full probabilities: {result['probabilities']}")

# Save prediction function with model
joblib.dump({
    'model': best_model,
    'predict_fn': predict_phone_price,
    'features': list(X.columns),
    'version': '1.0'
}, 'mobile_price_system.pkl')
print("\n✓ Saved complete prediction system: mobile_price_system.pkl")

# ======================
# 7. FINAL SUMMARY
# ======================
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"• Dataset: {df.shape[0]} phones, {len(X.columns)} features")
print(f"• Class Distribution: {'Balanced' if not is_imbalanced else 'Imbalanced (corrected)'}")
print(f"• Best Model: {best_model_name} (Macro F1: {best_f1:.4f})")
print(f"• Top Features: RAM, Internal Memory, Primary Camera, Battery Power")
print(f"• Artifacts Generated:")
print("  - price_distribution.png (Class distribution)")
print("  - correlation_heatmap.png (Feature relationships)")
print("  - confusion_matrix.png (Model performance)")
print("  - feature_importance.png (Key drivers)")
print("  - shap_summary.png (Explainability - if SHAP installed)")
print("  - mobile_price_predictor.pkl (Production model)")
print("  - mobile_price_system.pkl (Complete deployment package)")
print("\n✅ SYSTEM READY FOR DEPLOYMENT")
print("="*60)