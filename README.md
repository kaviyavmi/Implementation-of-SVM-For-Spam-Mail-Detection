# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KAVIYA V M
RegisterNumber: 212224040154 
*/
```

```PYTHON
# -----------------------------------------------
# STEP 1: Detect file encoding using chardet
# -----------------------------------------------
import chardet

# Define the file path
file = '/content/spam.csv'   # Path to the dataset (commonly in Colab environment)

# Open the file in binary mode to detect encoding
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))  # Read first 100,000 bytes and detect encoding

# Display detected encoding details
print("Detected encoding details:", result)

# -----------------------------------------------
# STEP 2: Load the dataset with correct encoding
# -----------------------------------------------
import pandas as pd

# Read the CSV using detected encoding
data = pd.read_csv(file, encoding=result['encoding'])
print("Data loaded successfully!")
print(data.head())

# -----------------------------------------------
# STEP 3: Separate input (X) and output (Y)
# -----------------------------------------------
x = data["v2"].values   # Usually the text messages (features)
y = data["v1"].values   # Usually the labels (spam / ham)

# -----------------------------------------------
# STEP 4: Split data into training and testing sets
# -----------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -----------------------------------------------
# STEP 5: Convert text data into numerical form
# -----------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorizer
cv = CountVectorizer()

# Learn vocabulary and transform training data (fit + transform)
x_train = cv.fit_transform(x_train)

# Transform test data (using same vocabulary)
x_test = cv.transform(x_test)

# -----------------------------------------------
# STEP 6: Train the Support Vector Machine model
# -----------------------------------------------
from sklearn.svm import SVC

# Create an instance of SVC (Support Vector Classifier)
svc = SVC()

# Train the model
svc.fit(x_train, y_train)

# -----------------------------------------------
# STEP 7: Make predictions on the test set
# -----------------------------------------------
y_pred = svc.predict(x_test)


# STEP 8: Evaluate the model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:

<img width="908" height="369" alt="image" src="https://github.com/user-attachments/assets/1d937c34-4c15-455a-bb57-c4692322c122" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
