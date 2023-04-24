import csv
import re
import random

def knn(dataset, query_point, num_of_iterations):
	pass


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


def remove_null(dataset):
    dado_limpo = []
    for row in dataset:
        # Verifica se a linha contém algum valor com "?"
        if re.search(r'\?', str(row.values())):
            continue
        dado_limpo.append(row)
    return dado_limpo


# Convert categorical attributes into dummy variables (one-hot encoding)
def one_hot_encoding(dataset, column):
    categories = {}
    for row in dataset:
        if row[column] not in categories:
            categories[row[column]] = len(categories)

    for row in dataset:
        category = row[column]
        row.pop(column)
        for i in range(len(categories)):
            row.insert(column + i, int(i == category))

"""
	Divides the data into a set of training data and a set of query data.
	Returns a tuple  (training_data, query_data)
"""
def divide_data(dataset):
	dataset_size = len(dataset)
	target_training_quantity = 0.7 * dataset_size
	training_data = random.choices(dataset, k=target_training_quantity)
	query_data = [row for row in dataset if row not in training_data]
	return training_data, query_data


def normalize_dataset(dataset):
	dataset = remove_null(dataset)
	max_values = {}
	for row in dataset:
		for key, value in row.items():
			try:
				value = float(value) #converte
			except ValueError: #caso falhe a conversao
				continue
			if key not in max_values or value > max_values[key]:
				max_values[key] = value
				# print(max_values[key]) #por curiosidade

	for row in dataset:
		for key, value in row.items():
			try:
				value = float(value)
			except ValueError:
				continue
			row[key] = value / max_values[key] #faz a divisao

	return dataset


data = read_dataset()
normalized_data = normalize_dataset(data)
training_data, query_data = divide_data(normalized_data)
# apenas para teste
with open('data/crx_normalized.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=normalized_data[0].keys())
    writer.writeheader()
    for row in normalized_data:
        writer.writerow(row)
