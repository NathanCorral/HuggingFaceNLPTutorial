#!/usr/bin/env python3
# https://huggingface.co/learn/nlp-course/chapter5/3
# Dataset Source:
#		https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
# 		https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com
# Data wget url: 
#		https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip
#	unzip drugsCom_raw.zip
import os

from datasets import load_dataset

PEAK_SAMPLES = False
VERBOSE = False


dataset_fold = f'{os.environ["HOME"]}/data/drugsCom'
data_files = {"train": f'{dataset_fold}/drugsComTrain_raw.tsv', \
				"test": f'{dataset_fold}/drugsComTest_raw.tsv'}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
"""
Quirks about the datast:  see source
	- ..
"""

if PEAK_SAMPLES:
	drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
	# Peek at the first few examples
	print(drug_sample[:3])


def assert_unique_column(data, column_name, split=None):
	"""
	Throw error on failure
	"""
	if split is None:
		for split in data.keys():
			# https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
			if len(data[split]) != len(data[split].unique(column_name)):
				raise ValueError(f'Non-unique split    {split}    column:    {column_name}')
				# raise ValueError(f'Non-unique split    {}    column:    {column_name}    is non-unique.')
	else:
		data_split = data
		if len(data_split) != len(data.unique(column_name)):
			raise ValueError(f'Non-unique column:    {column_name}')
	# return True


assert_unique_column(drug_dataset, "Unnamed: 0")
# verify_unique_column(drug_dataset, "rating") # raises ValueError

# Rename column
old = "Unnamed: 0"
new = "patient_id"
drug_dataset = drug_dataset.rename_column(
    original_column_name=old, new_column_name=new
)
if VERBOSE:
	print(drug_dataset)

"""
Try it out! 
Use the Dataset.unique() function to find the number of unique drugs 
	and conditions in the training and test sets.
"""
unique_drugs = {}
for split in ["train", "test"]:
	column_names = drug_dataset["train"].column_names

	unique_columns = []
	for column_name in column_names:
		try:
			assert_unique_column(drug_dataset[split], column_name, split=split)
			unique_columns.append(column_name)
		except ValueError as error:
			pass

	print(f'{split}')
	print(f'\tNumber Samples:  {len(drug_dataset[split]["drugName"])}')
	unique_drugs[split] = drug_dataset[split].unique("drugName")
	print(f'\tUnqiue drugs:  {len(unique_drugs[split])}')

	unique_drugs[split].sort()
	print(f'\tUnique columns:  {unique_columns}')
	# print(f'Unique drugs:  \n{unique_drugs[split]}')

	file_path = f'outputs/Chapter5_3_unique_drugs_split_{split}.txt'
	with open(file_path, 'w') as file:
		print(f'\tWriting file:  {file_path}')
		for item in unique_drugs[split]:
			file.write(item + "\n")



