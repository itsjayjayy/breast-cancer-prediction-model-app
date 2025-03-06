Breast Cancer Wisconsin Diagnostic Dataset
Dataset Overview
The Breast Cancer Wisconsin Diagnostic (BCD) dataset contains information about features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Each data point represents a tumor, and the features describe various aspects of the tumor's cell structure. The dataset is primarily used to classify breast tumors into two categories: malignant or benign.

Key Features:
Data Type: Numeric, Continuous
Attributes: 30 real-valued features describing characteristics of cell nuclei present in breast cancer biopsies.
Classes: Malignant (M) and Benign (B)
Dataset Size:
Number of instances: 569
Number of attributes: 32 (including the label column)
30 features + 1 label (M/B)
Data Source:
The dataset is available from the UCI Machine Learning Repository.

Dataset Description
The dataset is composed of a set of 30 numeric features that are computed from a digitized image of a breast mass. These features describe the characteristics of the cell nuclei present in the image, such as:

Radius (mean of distances from center to points on the perimeter)
Texture (standard deviation of gray-scale values)
Perimeter
Area
Smoothness (local variation in radius lengths)
Compactness (perimeter^2 / area - 1.0)
Concavity (severity of concave portions of the contour)
Concave points (number of concave portions of the contour)
Symmetry
Fractal dimension (coastline approximation - 1)
Label Column:
The label column classifies tumors as:

M: Malignant (cancerous)
B: Benign (non-cancerous)
Columns in the Dataset:
ID: Unique identifier for each sample.
Features (Mean): 30 real-valued features describing the characteristics of the cell nuclei.
Label: Class label indicating if the tumor is benign (B) or malignant (M).
Sample of the Data:
ID	Radius Mean	Texture Mean	Perimeter Mean	Area Mean	Smoothness Mean	...	Label
1	17.99	10.38	122.8	1001	0.1184	...	M
2	20.57	17.77	132.9	1326	0.08474	...	M
3	19.69	21.25	130.0	1203	0.1036	...	M
4	11.42	20.38	77.58	386	0.1186	...	B
5	20.29	14.34	135.1	1297	0.1005	...	M
Dataset Structure:
plaintext
Copy
- breast_cancer_wisconsin_diagnostic.csv
  - Features (30 attributes)
  - Class Label (Malignant / Benign)
Usage
Example Code:
python
Copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("breast_cancer_wisconsin_diagnostic.csv")

# Feature matrix and target variable
X = data.drop("Label", axis=1)
y = data["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
Acknowledgements
This dataset was contributed by Dr. William H. Wolberg, University of Wisconsin Hospitals, Madison.
The dataset can be found on the UCI Machine Learning Repository.
References
UCI Repository: Breast Cancer Wisconsin (Diagnostic) Data Set
Wolberg, W. H., & Mangasarian, O. L. (1990). "Multisurface method of pattern separation for medical diagnosis applied to breast cancer data." IEEE Transactions on Biomedical Engineering.
