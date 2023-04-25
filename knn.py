import csv
import re
import random
from copy import deepcopy
from math import sqrt


def knn(training_data:list[dict], query_point:dict, num_of_neighbors:int):
	'''
	A tuple where the first element is the euclidian distance between the query point and a training point
	and the second element is the index of the training point in the dataset
	'''
	def make_distance_tuple(y, ind_y) : return (euclidian_dist(query_point, y), ind_y)

	distances = list(
		map(
			make_distance_tuple,
			training_data,
			range(0, len(training_data))
		)
	)

	def get_distance(dist_tuple) : return dist_tuple[0]
	def get_point_index(dist_tuple) : return dist_tuple[1]
	def get_point_from_dataset(point_index) : return training_data[point_index]
	def get_class_of_point(point) : return point["a16"]

	sorted_distances = sorted(distances, key=get_distance, reverse=True)
	k_nearest_neighbors_index = map(get_point_index, sorted_distances[:num_of_neighbors])
	k_nearest_neighbors = map(get_point_from_dataset, k_nearest_neighbors_index)

	knn_classes = list(map(get_class_of_point, k_nearest_neighbors))
	plus_class_occurrences  = knn_classes.count("+")
	minus_class_occurrences  = knn_classes.count("-")
	return "+" if plus_class_occurrences > minus_class_occurrences else "-"


'''
Calculates the euclidian distance between two vectors (`a` and `b`)
'''
def euclidian_dist(a:dict, b:dict) -> float:
	cols = list(a)
	# Removing class column because its value is a string
	cols.remove("a16")
	sum_of_squared_diffs = 0
	for col in cols:
		comp_a = a[col]
		comp_b = b[col]
		sum_of_squared_diffs = sum_of_squared_diffs + ( (comp_a - comp_b) ** 2 )
	return sqrt(sum_of_squared_diffs)


'''
Returns a list of dicts corresponding to the dataset.

For example, from the following .csv:,

		first_name,last_name
		John, Cleese
		Terry, Gilliam

the first row of the dataset would look like this:

	{'first_name': 'John', 'last_name': 'Cleese'}

And the whole dataset would look like this:

	[
		{'first_name': 'John', 'last_name': 'Cleese'} ,
		{'first_name': 'Terry', 'last_name': 'Gilliam'}
	]

'''
def read_dataset():
	with open('data/crx.data', 'r') as file:
		reader = csv.DictReader(file)
		data = []
		for row in reader:
			data.append(row)
	return data

'''
Remove the dataset's NA (missing) values by looking which values are equal to `?`.
'''
def remove_null(dataset):
    dado_limpo = []
    for row in dataset:
        # Check if the line contains any value with "?"
        if re.search(r'\?', str(row.values())):
            continue
        dado_limpo.append(row)
    return dado_limpo

"""
Convert categorical attributes into dummy variables (one-hot encoding)
"""
def one_hot_encoding(dataset: dict, column: str):

	categories = set()

	# Discover categories
	for row in dataset:
		if row[column] not in categories:
			categories = categories | { row[column] }

	# create new column for each category discovered
	for row in dataset:
		for category in categories:
			new_col_name = f"{column}_{category}"
			value = row[column]
			row[new_col_name] = int(value == category)
		row.pop(column)

	return dataset


def  one_hot_encode_all_columns(dataset) :
	to_encode = deepcopy(dataset)
	one_hot_encoding(to_encode, "a1")
	one_hot_encoding(to_encode, "a4")
	one_hot_encoding(to_encode, "a5")
	one_hot_encoding(to_encode, "a6")
	one_hot_encoding(to_encode, "a7")
	one_hot_encoding(to_encode, "a9")
	one_hot_encoding(to_encode, "a10")
	one_hot_encoding(to_encode, "a12")
	one_hot_encoding(to_encode, "a13")
	return to_encode



"""
Divides the data into a set of training data (70%) and a set of query data (30%).
Returns a tuple  (training_data, query_data)
"""
def divide_data(dataset):
	dataset_size = len(dataset)
	target_training_quantity = int( 0.7 * dataset_size )
	training_data = random.choices(dataset, k=target_training_quantity)
	query_data = [row for row in dataset if row not in training_data]
	return training_data, query_data


"""
Normalizes the dataset by diving each value on a column by the maximum value of
that column found in the dataset
"""
def normalize_dataset(dataset):
	dataset = remove_null(dataset)
	max_values = {}
	for row in dataset:
		for key, value in row.items():
			try:
				value = float(value) #converts
			except ValueError: #in case of conversion failure
				continue
			if key not in max_values or value > max_values[key]:
				max_values[key] = value

	for row in dataset:
		for key, value in row.items():
			try:
				value = float(value)
			except ValueError:
				continue
			row[key] = value / max_values[key] #division

	return dataset

"""
Calculates the accuracy of running k-NN
"""
def accuracy(points, predicted_values) :
	num_of_points = len(points)
	true_predictions = 0

	for point, prediction in zip(points, predicted_values):
		# a16 is the column name which contain the classes categories.
		if point["a16"] == prediction:
			true_predictions = true_predictions + 1

	return true_predictions / num_of_points

"""
Runs k-NN on the `Credit Approval` dataset, making sure that before running:
	1 - All null data is removed;
	2 - Categorical data is one-hot encoded
	3 - All numeric values are normalized
	4 - k-NN is trained on 70% of the full dataset

After running the algorithm, it outputs to STDOUT the accuracy obtained from querying 30% of the data against the
training data and comparing expected classes X predicted classes.
"""
def main() :
	neighbors = 7
	data = read_dataset()
	normalized_data = normalize_dataset(data)
	encoded_data = one_hot_encode_all_columns(normalized_data)
	training_data, query_data = divide_data(encoded_data)

	predictions = []
	for query_point in query_data:
		predicted_class = knn(training_data, query_point, neighbors)
		predictions.append(predicted_class)
	accuracy_knn = accuracy(query_data, predictions)

	print(f"Accuracy of KNN with k={neighbors} is {accuracy_knn}")


if (__name__ == "__main__") : main()
