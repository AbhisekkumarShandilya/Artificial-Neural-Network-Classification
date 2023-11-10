from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data and pre-processing
data = pd.read_csv("car-groupC.csv")
data['Person'].fillna(2, inplace = True)
data['Maint_costs'].fillna('low', inplace = True)
le = LabelEncoder()
for column in ['Buy', 'Maint_costs', 'Doors', 'Person', 'Boot', 'Safety', 'Quality']:
    data[column] = le.fit_transform(data[column])
data.to_csv('Preprocessed_output.csv', index=False)


# Split the data into training and testing sets
X = data.iloc[:,0:-1]
y = data.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the classifier
clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=2000, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=21,tol=0.000000001)

# Train the classifier
clf.fit(X_train, y_train)

# Test the classifier
y_pred = clf.predict(X_test)

# Calculate metrics
loss_value = clf.loss_
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, average='macro')
uar = recall_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

print("Loss value: ", loss_value)
print("Accuracy: ", accuracy)
print("Sensitivity: ", sensitivity)
print("UAR: ", uar)
print("AUC: ", auc)


# Plot loss curve
plt.plot(clf.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index = ['actual-unacc', 'actual-acc', 'actual-good', 'actual-vgood'], 
                     columns = ['predict-unacc', 'predict-acc', 'predict-good', 'predict-vgood'])
print("Confusion Matrix: ")
print(cm_df)


