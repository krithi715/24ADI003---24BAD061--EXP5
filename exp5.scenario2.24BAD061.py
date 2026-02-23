import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

features = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 
            'Education', 'Property_Area']

X = df[features].copy()
y = df['Loan_Status'].copy()

X['LoanAmount'] = X['LoanAmount'].fillna(X['LoanAmount'].median())
X['Credit_History'] = X['Credit_History'].fillna(X['Credit_History'].mode()[0])
X['Education'] = X['Education'].fillna(X['Education'].mode()[0])
X['Property_Area'] = X['Property_Area'].fillna(X['Property_Area'].mode()[0])
y = y.fillna(y.mode()[0])

le_edu = LabelEncoder()
le_prop = LabelEncoder()
le_target = LabelEncoder()

X.loc[:, 'Education'] = le_edu.fit_transform(X['Education'])
X.loc[:, 'Property_Area'] = le_prop.fit_transform(X['Property_Area'])
y = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_shallow.fit(X_train, y_train)

tree_deep = DecisionTreeClassifier(random_state=42)
tree_deep.fit(X_train, y_train)

y_pred_shallow = tree_shallow.predict(X_test)
y_pred_deep = tree_deep.predict(X_test)

print("Shallow Tree Accuracy:", accuracy_score(y_test, y_pred_shallow))
print("Deep Tree Accuracy:", accuracy_score(y_test, y_pred_deep))

print("\nShallow Tree Classification Report:\n")
print(classification_report(y_test, y_pred_shallow))

print("\nDeep Tree Classification Report:\n")
print(classification_report(y_test, y_pred_deep))

cm = confusion_matrix(y_test, y_pred_shallow)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=le_target.classes_)
disp.plot()
plt.title("Confusion Matrix - Decision Tree (Max Depth = 3)")
plt.xlabel("Predicted Loan Status")
plt.ylabel("Actual Loan Status")
plt.show()

plt.figure(figsize=(14,8))
plot_tree(tree_shallow, 
          feature_names=features,
          class_names=le_target.classes_,
          filled=True)
plt.title("Decision Tree Structure (Max Depth = 3)")
plt.xlabel("Tree Splits Based on Features")
plt.ylabel("Tree Depth Levels")
plt.show()

importances = tree_shallow.feature_importances_

plt.figure()
plt.bar(features, importances, label="Importance Score")
plt.title("Feature Importance in Loan Prediction")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.legend()
plt.xticks(rotation=45)
plt.show()

train_acc_shallow = accuracy_score(y_train, tree_shallow.predict(X_train))
test_acc_shallow = accuracy_score(y_test, y_pred_shallow)

train_acc_deep = accuracy_score(y_train, tree_deep.predict(X_train))
test_acc_deep = accuracy_score(y_test, y_pred_deep)

print("Shallow Tree Train Accuracy:", train_acc_shallow)
print("Shallow Tree Test Accuracy:", test_acc_shallow)
print("Deep Tree Train Accuracy:", train_acc_deep)
print("Deep Tree Test Accuracy:", test_acc_deep)