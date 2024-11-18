from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


#creating a data set

data = {

    "feature1": [1, 2, 3, 4, 5,],
    "feature2": [5, 4, 3, 2, 1], 
    "Label": [0, 1, 0 , 1, 0],
}

#convert the data set into a pands dataframe

df = pd.DataFrame(data)

X = df[["feature1", "feature2"]] #features
y = df[["Label"]] #labels

#Features are the data the model uses to learn. for example

'''
          [1, 2, 3, 4, 5,]
    and   [5, 4, 3, 2, 1]
    
    help the model figure out patterns

    we use double brackets to tell pythton to select multiple columns and return them as a dataframe
'''

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=42)

#test-size means 20% of the data will be used for testing, and 80% will be used for training

#random_state ensure that the split is reproducible. Without this, the split would be random every time you run the code, and results could vary

classifier = DecisionTreeClassifier(max_depth=3)

classifier.fit(X_train, y_train)

'''
Fit is used to find patterns in the training data, 
X_train and matching the labels (y_train)

the decision tree learns to split the data bsed on the features to classify it correctly

'''

y_pred = classifier.predict(X_test)

'''
this uses the trained model(our clssifier) to predict labels for unseen dest data (X_test)

how it works: The decision tree applies the rules it learned during training to classify each row in X_test

y_pred stores the predicted labels for the test dta. These predictions can then be compared with the actual labels (y_test) to see how accurate the model is
'''

accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

#plotting the decision tree
plt.figure(figsize=(6, 10))  # Make the plot larger for better readability
plot_tree(
    classifier, 
    feature_names=['Feature1', 'Feature2'], 
    class_names=['Class 0', 'Class 1'], 
    filled=True, 
    fontsize=6,  # Increase the font size for readability
    rounded=True  # Use rounded boxes for better visuals
) #set the size of the plot

'''
Understanding the Text in the Boxes
Each box in the tree represents a decision node or a leaf node. Here’s what the text means:

FeatureX <= Value:

This is the decision rule applied at the node.
Example: Feature2 <= 2.5 means the data is split based on whether Feature2 is less than or equal to 2.5.
gini:

The Gini Impurity is a measure of how mixed the classes are at this node.
A gini of 0 means all samples at the node belong to one class (perfectly pure), while a higher gini indicates more mixing.
samples:

The number of data points (rows) that reach this node.
Example: samples = 4 means 4 data points reach this decision.
value:

The count of data points for each class.
Example: value = [3, 1] means 3 data points belong to Class 0, and 1 data point belongs to Class 1.
class:

The predicted class at this node.
This is the class with the majority count in value.
The plot_tree function visualizes the decision tree, showing the rules at each node,
the Gini impurity, and the number of samples for each split.
'''


print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plot_tree(classifier, feature_names=["Feature1", "Feature2"], class_names=["Class 0", "Class 1"], filled=True)
plt.show()