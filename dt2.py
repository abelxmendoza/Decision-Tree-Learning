import pandas as pd

def learn_decision_tree_recursive(X, y, depth=0):
    print(f"Step {depth}: Data points: {len(y)}")
    print(f"Decision: y = {y.iloc[0]} (Leaf Node)\n")

def learn_decision_tree(data):
    X = data[['A1', 'A2', 'A3']]
    y = data['Output y']
    learn_decision_tree_recursive(X, y)

def main():
    data = pd.DataFrame({
        'A1': [0, 0, 1, 0, 0],
        'A2': [0, 0, 1, 1, 1],
        'A3': [0, 1, 1, 0, 1],
        'Output y': [0, 0, 0, 1, 1]
    })

    learn_decision_tree(data)

if __name__ == "__main__":
    main()

