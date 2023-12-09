# decision_tree_learning.py

from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import matplotlib.pyplot as plt

def learn_decision_tree(data):
    X = data[['A1', 'A2', 'A3']]
    y = data['Output y']
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y)
    return decision_tree

def main():
    data = pd.DataFrame({
        'A1': [0, 0, 1, 0, 0],
        'A2': [0, 0, 1, 1, 1],
        'A3': [0, 1, 1, 0, 1],
        'Output y': [0, 0, 0, 1, 1]
    })

    tree_model = learn_decision_tree(data)

    # Visualize the decision tree with correct feature names
    plt.figure(figsize=(10, 7))
    plot_tree(tree_model, filled=True, feature_names=['A1', 'A2', 'A3'], class_names=['0', '1'], rounded=True, precision=2)
    plt.show()

if __name__ == "__main__":
    main()
