#!/usr/bin/env python3
# Part 1:  https://huggingface.co/learn/nlp-course/en/chapter3/2
# Part 2:  https://huggingface.co/learn/nlp-course/en/chapter3/3?fw=pt

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# Config data
checkpoint = "bert-base-uncased"
auto_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
auto_tokenizer_settings = {"truncation": True}

# Program Data
# Glue secitons
glue_subsections = ["ax",
					"cola",
					"mnli",
					"mnli_matched",
					"mnli_mismatched",
					"mrpc",
					"qnli",
					"qqp",
					"rte",
					"sst2",
					"stsb",
					"wnli",
					]

glue_subsections_string_keys = {
	"ax": ["premise", "hypothesis"],
	"cola": ["sentence"],
	"mnli": ["premise", "hypothesis"],
	"mnli_matched": ["premise", "hypothesis"],
	"mnli_mismatched": ["premise", "hypothesis"],
	"mrpc": ["sentence1", "sentence2"],
	"qnli": ["question", "sentence"],
	"qqp": ["question1", "question2"],
	"rte": ["sentence1", "sentence2"],
	"sst2": ["sentence"],
	"stsb": ["sentence1", "sentence2"],
	"wnli": ["sentence1", "sentence2"]
}

# These keys contain human-readable semantic data.
#	They do not need to be collated during batch assembly and batched to the same length
#	but are still present in the dataset
semantic_keynames = ["idx", 
					"sentence", "sentence1", "sentence2", 
					"question", "question1", "question2",
					"premise", "hypothesis"
]


def print_data(raw_datasets, glue_subsection=None):
	# print(f"Raw Datasets:   {raw_datasets.keys()}")
	ex_split = list(raw_datasets.keys())[0]
	assert ex_split in ["train", "validation", "test"]

	ex_dataset = raw_datasets[ex_split]

	# print(f'Dataset Features:  {ex_dataset.features}')
	# print(f'Dataset Features[label]:  {ex_dataset.features["label"]}')
	if hasattr(ex_dataset.features["label"], "names"):
		label_lookup = ex_dataset.features["label"].names
	else:
		# The "stsb" subsection only has a float as a label.  No label names  
		class CustomDict(dict):
			"""
			Create a class which is an empty dict
			Whenever we try to look up a "labal"'s name, it returns just the label
				ex. print(label_lookup[2]) -> 2
				ex. print(label_lookup[2.34324]) -> 2.34324
			We only need this for glue "stsb" subsection, which characterizes the similarity of
				sentences with a float in [0, 5]
			"""
			def __missing__(self, key):
				return key
		label_lookup = CustomDict()


	dataset_strs = [ex_dataset[key] for key in glue_subsections_string_keys[glue_subsection]]
	for i in range(3):
		label = ex_dataset['label'][i]
		label_str = label_lookup[label]

		print_str = f'[{i:04d}] Label: \033[1m{label_str}\033[0m'
		sentence_strs = [f'\n  -{k}: {s[i]}' for k, s in zip(glue_subsections_string_keys[glue_subsection], dataset_strs)]
		print_str = ''.join([print_str] + sentence_strs)

		print(print_str)

	print("\n")


def pre_process_glue(raw_datasets, subsection="sst2", debug_print=False):
	assert subsection in glue_subsections

	if debug_print:
		print_data(raw_datasets, glue_subsection=subsection)

	def tokenize_funcion(example):
		example_sentences = [example[key] for key in glue_subsections_string_keys[subsection]]
		# print(example_sentences)
		return auto_tokenizer(*example_sentences, **auto_tokenizer_settings)
	tokenized_datasets = raw_datasets.map(tokenize_funcion, batched=True)

	# Default Hugging Face individual batch padding
	data_collator = DataCollatorWithPadding(tokenizer=auto_tokenizer)
	return tokenized_datasets, data_collator



if __name__ == "__main__":
	############################################
	# Part 2
	############################################
	for subsection in glue_subsections:
		print("="*25)
		print(f"Subsection:  {subsection}")

		raw_datasets = load_dataset("glue", subsection)
		# Add tokenize and collate functions
		tokenized_datasets, data_collator = pre_process_glue(raw_datasets, subsection=subsection)
		

		# Example Batch that can be passed to model:
		ex_split = list(raw_datasets.keys())[0]
		assert ex_split in ["train", "validation", "test"]
		samples = tokenized_datasets[ex_split][:8]
		samples = {k: v for k, v in samples.items() if k not in semantic_keynames}
		batch = data_collator(samples)
		print({k: v.shape for k, v in batch.items()})
		print("\n")

	############################################
	# Part 2
	############################################




