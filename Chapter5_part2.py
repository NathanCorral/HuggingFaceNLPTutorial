#!/usr/bin/env python3
# https://huggingface.co/learn/nlp-course/chapter5/2

import os
from datasets import load_dataset


dataset_fold = f'{os.environ["HOME"]}/data/SQuAD-it'
data_files = {"train": f'{dataset_fold}/SQuAD_it-train.json', "test": f'{dataset_fold}/SQuAD_it-test.json'}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

print(squad_it_dataset)

# Do the same thing but from remote
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset_remote = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset_remote)
