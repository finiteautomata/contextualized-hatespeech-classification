"""
Script to train hatespeech classifier
"""
import os
import fire
import json
import tempfile
import torch
import sys
import random
from transformers import (
    TrainingArguments, DataCollatorWithPadding, Trainer
)
from hatedetection import load_datasets, extended_hate_categories
from hatedetection.training import (
    tokenize, lengths, load_model_and_tokenizer,
    lengths, train_finegrained
)
from hatedetection.metrics import compute_extended_category_metrics



def run_finegrained_experiments(
    output_path, times, model_name, context,
    batch_size=32, eval_batch_size=32, output_dir=None,
    accumulation_steps=1, max_length=None, epochs=5, warmup_ratio=0.1,
    use_class_weight=False, use_dynamic_padding=True,
    ):

    """
    Train fine-grained classifier

    Arguments:
    ----------

    train_path:
        Path to training data
    test_path:
        Path to test data

    Returns:
    --------

    trainer: transformers.Trainer
    """
    print("*"*80)
    print(f"Training hate speech fine grained classifier -- {output_path}")


    results = {
        "model_name": model_name,
        "use_class_weight": use_class_weight,
        "context": context,
        "times": times,
        "metrics": [],
        "labels": [],
        "predictions": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(times):
        print(("="*80+'\n')*3)
        print(f"{i+1} iteration", "\n"*3)


        seed = 20212020 - i
        random.seed(seed)
        output_dir = tempfile.TemporaryDirectory().name
        model, tokenizer = load_model_and_tokenizer(
            model_name, num_labels=len(extended_hate_categories), device=device,
            max_length=lengths[context],
        )

        add_body = True if "body" in context else False

        train_dataset, dev_dataset, test_dataset = load_datasets(add_body=add_body)

        trainer, test_dataset = train_finegrained(
            model, tokenizer, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
            context=context, batch_size=batch_size, eval_batch_size=eval_batch_size,
            accumulation_steps=accumulation_steps, epochs=epochs,
            use_class_weight=use_class_weight,
            warmup_ratio=warmup_ratio,
            output_dir=output_dir
        )

        eval_training_args = TrainingArguments(
            output_dir=".",
            per_device_eval_batch_size=eval_batch_size,
        )


        eval_trainer = Trainer(
            model=trainer.model,
            args=eval_training_args,
            compute_metrics=lambda pred: compute_extended_category_metrics(test_dataset, pred),
            data_collator = DataCollatorWithPadding(tokenizer, padding="longest") if use_dynamic_padding else None,
        )

        preds = eval_trainer.predict(test_dataset)

        results["labels"].append(preds.label_ids.tolist())
        results["predictions"].append(preds.predictions.tolist())
        results["metrics"].append(preds.metrics)

        with open(output_path, "w+") as f:
            json.dump(results, f, indent=4)


        os.system(f"rm -Rf {output_dir}")


    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    fire.Fire(run_finegrained_experiments)

