"""
Script to train hatespeech classifier
"""
import json
import sys
import fire
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from hatedetection import load_datasets, extended_hate_categories
from hatedetection.model import BertForSequenceMultiClassification
from hatedetection.training import (
    tokenize, lengths
)
from hatedetection.metrics import compute_extended_category_metrics
from sklearn.metrics import precision_recall_fscore_support





def eval_hate_category(
    model_name, output_path=None, context='none', eval_batch_size=16, use_all=False, dev=False,
    ):
    """
    Evaluates a model
    """
    if context not in lengths.keys():
        print(f"context must be one of {lengths.keys()}")
        sys.exit(1)

    print(f"Model name: {model_name}")
    print("Context: ", context)
    _, dev_dataset, test_dataset = load_datasets(add_body=True)

    if not use_all:
        dev_dataset = dev_dataset.filter(lambda x: x["HATEFUL"] > 0)
        test_dataset = test_dataset.filter(lambda x: x["HATEFUL"] > 0)

    dataset = test_dataset if not dev else dev_dataset
    model = BertForSequenceMultiClassification.from_pretrained(model_name, num_labels=len(extended_hate_categories))

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Tokenizing and formatting \n\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = lengths[context]

    dataset = dataset.map(
        lambda x: tokenize(tokenizer, x, context=context), batched=True, batch_size=eval_batch_size
    )

    def format_dataset(dataset):
        def get_category_labels(examples):
            return {
                'labels': torch.Tensor(
                    [examples[cat] for cat in extended_hate_categories]),
            }
        dataset = dataset.map(get_category_labels)
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
        compute_metrics=lambda pred: compute_extended_category_metrics(dataset, pred),
    )

    preds = trainer.predict(dataset)

    serialized = {
        "predictions": preds.predictions.tolist(),
        "labels": preds.label_ids.tolist(),
        "metrics": preds.metrics
    }

    for k, v in serialized["metrics"].items():
        print(f"{k:<15} =  {v:.4f}")

    if output_path:
        print(f"Saving at {output_path}")

        with open(output_path, "w+") as f:
            json.dump(serialized, f, indent=4)



if __name__ == '__main__':
    fire.Fire(eval_hate_category)
