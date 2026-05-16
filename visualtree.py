from sklearn.tree import export_text
import pandas as pd
import joblib

# Load model
rf = joblib.load("final_ddos_rf_model.pkl")

# Load features safely
df = pd.read_csv("final_features.csv")
features = list(df.iloc[:, 0])   # FIXED

# Get one tree
tree = rf.estimators_[0]

# Print rules
print(export_text(tree, feature_names=features))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=features, filled=True, max_depth=3)
plt.show()