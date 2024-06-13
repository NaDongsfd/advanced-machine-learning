from math import sqrt

# 1. Calculate the Euclidean distance between two rows
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):  # exclude the last element which is the label
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# 2. Locate the k most similar neighbors
def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# 3. Make a prediction with k neighbors
def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Example usage with specific 2D data
dataset = [
    [2, 3, 0], [3, 3, 0], [3, 2, 0],
    [5, 8, 1], [6, 8, 1], [6, 9, 1],
    [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [7, 7, 1]
]

# Test point
test_point = [4, 4]

# Number of neighbors
k = 3

# Predict the classification of the test point
predicted_class = predict_classification(dataset, test_point, k)
print(f'The test point {test_point} belongs to class {predicted_class}')