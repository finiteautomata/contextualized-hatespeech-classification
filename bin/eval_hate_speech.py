"""
Script to train hatespeech classifier
"""
import json
import sys
import fire
import torch
from hatedetection import load_datasets
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from hatedetection.training import (
    tokenize, lengths
)
from hatedetection.metrics import compute_hate_metrics


def eval_hate_speech(
    model_name, output_path, context='none', eval_batch_size=16,
    ):
    """
    Evaluates a model
    """
    if context not in lengths.keys():
        print(f"context must be one of {lengths.keys()}")
        sys.exit(1)

    print(f"Model name: {model_name}")
    print("Context: ", context)
    _, _, test_dataset = load_datasets(add_body=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=2)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Tokenizing and formatting \n\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = lengths[context]
    dataset = test_dataset

    dataset = dataset.map(
        lambda x: tokenize(tokenizer, x, context=context), batched=True, batch_size=eval_batch_size
    )

    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['HATEFUL']})
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset

    dataset = format_dataset(dataset)

    print("Sanity check\n\n")

    print(tokenizer.decode(dataset[0]["input_ids"]), "\n\n")

    print("Predicting")

    training_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=eval_batch_size,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_hate_metrics,
    )

    preds = trainer.predict(dataset)

    serialized = {
        "predictions": preds.predictions.tolist(),
        "labels": preds.label_ids.tolist(),
        "metrics": preds.metrics
    }

    print(f"Saving at {output_path}")

    with open(output_path, "w+") as f:
        json.dump(serialized, f, indent=4)



if __name__ == '__main__':
    fire.Fire(eval_hate_speech)

