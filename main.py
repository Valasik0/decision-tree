from DecisionTree import *
from sklearn.model_selection import train_test_split


data = pd.read_csv('data_classification/data_classification/titanic_preprocessed.csv', delimiter=',')
train_data, test_data = train_test_split(data, test_size=0.2)

for i in range(1, 16):
    tree = DecisionTree(max_depth=i, target_column_index=1)
    tree.fit(train_data)
    score = tree.calculate_score(test_data)
    print(f"Max depth: {i}, Skore: {score}")

"""

data = pd.read_csv('data_classification/data_classification/iris.csv', delimiter=';')

train_data, test_data = train_test_split(data, test_size=0.2)

tree = DecisionTree(max_depth=5)
tree.fit(train_data)

tree.draw_tree()
score = tree.calculate_score(test_data)

print(f"Skore: {score:.2f}")

tree_prediction = tree.predict_row(test_data.iloc[0])
print(f"Predikovaná hodnota: {tree_prediction}")

real_value = test_data.iloc[0].iloc[-1]
print(f"Reálná hodnota {int(real_value)}")
"""