import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(model, X_val, y_val, filename='confusion_matrix.png'):
    """
    Generates, plots, and saves a confusion matrix for performance analysis.
    """
    print("🎨 Generating Confusion Matrix plot...")
    
    # 1. Generate predictions
    y_pred = model.predict(X_val)
    
    # 2. Compute the matrix values
    cm = confusion_matrix(y_val, y_pred)
    
    # 3. Create the plot using Seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate (0)', 'Fraudulent (1)'],
                yticklabels=['Legitimate (0)', 'Fraudulent (1)'])
    
    plt.title('XGBoost Fraud Detection Confusion Matrix', fontsize=12, pad=15)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    
    # 4. Save the plot into your project root folder
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"✅ Performance plot saved successfully as '{filename}'!")
