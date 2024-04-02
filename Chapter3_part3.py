#!/usr/bin/env python3
# Part 3:  https://huggingface.co/learn/nlp-course/en/chapter3/3?fw=pt
import numpy as np

# Previous work
from datasets import load_dataset
from Chapter3_part2 import pre_process_glue, semantic_keynames, auto_tokenizer

# New imports
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification

import evaluate

checkpoint = "bert-base-uncased"
subsection = "mrpc"
subsection = "sst2"

glue_labels_per_subsection = {
	"sst2": 2, # Binary classification uses 2 labels
	"mrpc": 2
}


metric = evaluate.load("glue", subsection)
def compute_metrics(eval_preds):
	logits, labels = eval_preds
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


def glue_evaluate(trainer, glue_subsection, dataset):
	glue_metric = evaluate.load("glue", glue_subsection)

	predictions = trainer.predict(dataset)
	print(predictions.predictions.shape, predictions.label_ids.shape)
	print("predictction metrics:  ",  predictions.metrics)

	preds = np.argmax(predictions.predictions, axis=-1)
	result = glue_metric.compute(predictions=preds, references=predictions.label_ids)

	return result


def try_load_model_tokenizer(ckpt_path):
	model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

	pass



if __name__ == "__main__":
	# Last Part:
	raw_datasets = load_dataset("glue", subsection)
	tokenized_datasets, data_collator = pre_process_glue(raw_datasets, subsection)


	# Example batch
	# ex_split = list(raw_datasets.keys())[0]
	# assert ex_split in ["train", "validation", "test"]
	# samples = tokenized_datasets[ex_split][:8]
	# samples = {k: v for k, v in samples.items() if k not in semantic_keynames}
	# batch = data_collator(samples)
	# print({k: v.shape for k, v in batch.items()})
	# print("\n")


	# Part 3.
	# training_args = TrainingArguments(f'test-trainer-{subsection}')
	training_args = TrainingArguments(
		    output_dir=f'./test-trainer-{subsection}',          
		    								 # Directory where the model predictions and checkpoints will be written.
		    num_train_epochs=3,              # Total number of training epochs to perform.
		    per_device_train_batch_size=8,   # Batch size per device during training.
		    per_device_eval_batch_size=8,    # Batch size for evaluation.
		    warmup_steps=500,                # Number of warmup steps for learning rate scheduler.
		    weight_decay=0.01,               # Strength of weight decay.
		    logging_dir='./logs',            # Directory for storing logs.
		    logging_steps=10,                # Log every X updates steps.
		    evaluation_strategy="steps",     # Evaluation is done (and logged) every `eval_steps`.
		    eval_steps=100,                  # Evaluation happens every 100 steps.
		    load_best_model_at_end=True,     # Load the best model when finished training (as measured by `metric_for_best_model`).
		    metric_for_best_model="accuracy",# Use accuracy to evaluate the best model.
		    learning_rate=5e-05,
	)

	num_labels = glue_labels_per_subsection[subsection]
	print("num_labels:  ",  num_labels)
	model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
	print("\n\nModel: ")
	print(model)

	trainer = Trainer(
	    model,
	    training_args,
	    train_dataset=tokenized_datasets["train"],
	    eval_dataset=tokenized_datasets["validation"],
	    data_collator=data_collator,
	    tokenizer=auto_tokenizer,
	    compute_metrics=compute_metrics,
	)
	trainer.train()

	# Final evaluation
	result = glue_evaluate(trainer, subsection, tokenized_datasets["validation"])
	print("Result:  ")
	print(result)