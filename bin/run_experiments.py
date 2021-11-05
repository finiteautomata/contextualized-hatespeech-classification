"""
Script to train hatespeech classifier
"""
import os
import fire
import json
import time
import tempfile
import torch
from transformers import (
    TrainingArguments, DataCollatorWithPadding, Trainer, set_seed
)
from hatedetection import load_datasets, extended_hate_categories
from hatedetection.training import (
    lengths, load_model_and_tokenizer,
    lengths, train_classifier
)
from hatedetection.metrics import compute_extended_category_metrics



def run_experiments(
    output_path, times, model_name, context,
    batch_size=32, output_dir=None,
    accumulation_steps=1, max_length=None, epochs=5, warmup_ratio=0.1,
    use_class_weight=False, use_dynamic_padding=True,
    plain=False,
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


    if not os.path.exists(output_path):
        results = {
            "model_name": model_name,
            "use_class_weight": use_class_weight,
            "context": context,
            "times": times,
            "metrics": [],
            "labels": [],
            "predictions": [],
        }
    else:
        with open(output_path, "r") as f:
            results = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    times_left = times - len(results["metrics"])

    print(f"We have to run this for {times_left} times")

    for i in range(times_left):
        print(("="*80+'\n')*3)
        print(f"{i+1} iteration", "\n"*3)


        set_seed(int(time.time()))
        output_dir = tempfile.TemporaryDirectory().name

        num_labels = 2 if plain else len(extended_hate_categories)

        model, tokenizer = load_model_and_tokenizer(
            model_name, num_labels=num_labels, device=device,
            max_length=lengths[context],
        )

        add_body = True if "body" in context else False

        train_dataset, dev_dataset, test_dataset = load_datasets(add_body=add_body)

        trainer, test_dataset = train_classifier(
            model, tokenizer, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
            context=context, batch_size=batch_size, eval_batch_size=batch_size,
            accumulation_steps=accumulation_steps, epochs=epochs,
            use_class_weight=use_class_weight,
            warmup_ratio=warmup_ratio, output_dir=output_dir,
            plain=plain,
        )


        if not plain:
            eval_training_args = TrainingArguments(
                output_dir=".",
                per_device_eval_batch_size=batch_size,
            )
            eval_trainer = Trainer(
                model=trainer.model,
                args=eval_training_args,
                compute_metrics=lambda pred: compute_extended_category_metrics(test_dataset, pred),
                data_collator = DataCollatorWithPadding(tokenizer, padding="longest") if use_dynamic_padding else None,
            )
        else:
            eval_trainer = trainer
        preds = eval_trainer.predict(test_dataset)
        results["labels"].append(preds.label_ids.tolist())
        results["predictions"].append(preds.predictions.tolist())
        results["metrics"].append(preds.metrics)

        with open(output_path, "w+") as f:
            json.dump(results, f, indent=4)


        os.system(f"rm -Rf {output_dir}")


    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    fire.Fire(run_experiments)

