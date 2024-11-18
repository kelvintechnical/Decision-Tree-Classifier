

# <h1>Decision Tree Classifier</h1>

<p>Welcome to the <strong>Decision Tree Classifier</strong> project! This repository contains code to implement a basic machine learning classifier using the Decision Tree algorithm. A decision tree classifies data points by splitting the data into branches based on feature values and ultimately assigning a label to each point.</p>

---

## <h2>📫 How to reach me:</h2>

<ul>
  <li><strong>Email:</strong> <a href="mailto:ktobia10@wgu.edu">ktobia10@wgu.edu</a></li>
  <li><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/kelvin-r-tobias-211949219/">Kelvin R. Tobias</a></li>
  <li><strong>Bluesky:</strong> <a href="https://bsky.app/profile/kelvintechnical.bsky.social">@kelvintechnical.bsky.social</a></li>
  <li><strong>Instagram:</strong> <a href="https://www.instagram.com/kelvinintech/">@kelvinintech</a></li>
</ul>

---

## <h2>Project Overview</h2>

<p>The <strong>Decision Tree Classifier</strong> is a simple and effective classification algorithm. In this project, we use Python libraries like <code>pandas</code>, <code>scikit-learn</code>, and <code>matplotlib</code> to build and visualize the decision tree. The goal is to classify data points based on specific features and evaluate the model's accuracy.</p>

---

## <h2>5 Things I Learned from This Project</h2>

<ol>
  <li><strong>The Purpose of Imports:</strong> Each Python library has specific functionalities that simplify machine learning workflows.</li>
  <li><strong>Data Splitting:</strong> Dividing data into training and testing sets ensures the model can generalize to unseen data.</li>
  <li><strong>Model Visualization:</strong> Plotting the decision tree helped me understand how the algorithm splits data at each node.</li>
  <li><strong>Parameter Tuning:</strong> Adjusting parameters like <code>max_depth</code> can simplify the model and reduce overfitting.</li>
  <li><strong>Evaluation Metrics:</strong> Accuracy alone might not always be enough, and exploring additional metrics like confusion matrices can help evaluate models better.</li>
</ol>

---

## <h2>Code Explanation</h2>

<p>Below is an overview of the key components of the code:</p>

<ul>
  <li><code>from sklearn.tree import DecisionTreeClassifier, plot_tree</code>: Imports the Decision Tree Classifier for building the model and the plotting function for visualizing the tree.</li>
  <li><code>from sklearn.model_selection import train_test_split</code>: Splits the dataset into training and testing sets.</li>
  <li><code>from sklearn.metrics import accuracy_score</code>: Calculates the model's accuracy by comparing predictions to actual labels.</li>
  <li><code>import pandas as pd</code>: Used for handling the dataset as a DataFrame for easier data manipulation.</li>
  <li><code>import matplotlib.pyplot as plt</code>: Visualizes the decision tree with a readable plot.</li>
</ul>

```python
# Importing necessary libraries
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Create a simple dataset
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [5, 4, 3, 2, 1],
    'Label': [0, 1, 0, 1, 0]
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Separate the features (X) and labels (y)
X = df[['Feature1', 'Feature2']]  # Features
y = df['Label']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree Classifier
classifier = DecisionTreeClassifier(max_depth=3)

# Train the model using the training data
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))  # Adjust the plot size for readability
plot_tree(
    classifier, 
    feature_names=['Feature1', 'Feature2'], 
    class_names=['Class 0', 'Class 1'], 
    filled=True, 
    fontsize=10, 
    rounded=True
)
plt.show()
