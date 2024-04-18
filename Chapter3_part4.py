#!/usr/bin/env python3
# Part 4:  https://huggingface.co/learn/nlp-course/chapter3/4
from Chapter3_part2 import pre_process_glue, semantic_keynames, \
                            auto_tokenizer, glue_subsections_string_keys
from datasets import load_dataset

# Optimizer, lr scheduler, dataloader, model
from transformers import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

import torch
from tqdm.auto import tqdm

# Evaluation
import evaluate

# Change here for extra work:
checkpoint = "bert-base-uncased"
subsection = "mrpc"

glue_labels_per_subsection = {
    "sst2": 2, # Binary classification uses 2 labels
    "mrpc": 2
}

raw_datasets = load_dataset("glue", subsection)
tokenized_datasets, data_collator = pre_process_glue(raw_datasets, subsection)

"""
To fully prepare the dataset, we need to perfrom some preprocessing done automatically
by the Trainer object, namely we must:
    - Remove the columns corresponding to values the model does not expect
    - Rename the column 'label' to 'labels' (because the model expects this name)
    - Set the format of the datasets so they return PyTorch tensors
"""
columns_to_remove = [x for x in tokenized_datasets["train"].column_names if x in semantic_keynames]
print("Semantic Columns Removed:  ",  columns_to_remove)
tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# May depend on task we are fine-tuning for
expected_model_input_columns = ["attention_mask", "input_ids", "labels", "token_type_ids"]
def checkEqual(L1, L2):
    """
    https://stackoverflow.com/questions/12813633/how-to-assert-two-list-contain-the-same-elements-in-python
    """
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)
assert checkEqual(tokenized_datasets["train"].column_names, expected_model_input_columns)


# Create the dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Double check successful dataloaders:
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
#       Shapes may be different, since we shuffle and pad to max sentence length of batch



model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)



# Make sure it all works
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)





optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(f'num_training_steps:  {num_training_steps}')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(f'device:  {device}')

progress_bar = tqdm(range(num_training_steps), desc="Train")

model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = evaluate.load("glue", subsection)
model.eval()

progress_bar = tqdm(range(len(eval_dataloader)), desc="Eval")
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)

result = metric.compute()
print("\n\nResult:  ")
print(result)