"""
Script to train hatespeech classifier
"""
import os
import fire
import tempfile
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



def train_finegrained_classifier(
    output_path, train_path=None, test_path=None, context='none',
    model_name = 'dccuchile/bert-base-spanish-wwm-cased',
    batch_size=32, eval_batch_size=32, output_dir=None,
    accumulation_steps=1, max_length=None, epochs=5, warmup_ratio=0.1,
    random_seed=2021, use_class_weight=False, use_dynamic_padding=True,
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

    random.seed(random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    if context not in lengths.keys():
        print(f"{context} must be in {lengths.keys()}")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(
        model_name, num_labels=len(extended_hate_categories), device=device,
        max_length=lengths[context],
    )

    if not output_dir:
        output_dir = tempfile.TemporaryDirectory().name

    print(f"Uses context: {context}")
    print(f"Tokenizer max length: {max_length}")
    #print(f"Negative examples: {negative_examples_proportion}")
    print(f"Results dir: {output_dir}")

    print("*"*80, end="\n"*3)

    print("Loading datasets... ", end="")


    add_body = True if "body" in context else False

    train_dataset, dev_dataset, test_dataset = load_datasets(
        train_path, test_path, add_body=add_body
    )


    """
    # Don't do this for the time being

    if negative_examples_proportion is None:

        #Only train and test on negative examples
        train_dataset = train_dataset.filter(lambda x: x["HATEFUL"] > 0)
        dev_dataset = dev_dataset.filter(lambda x: x["HATEFUL"] > 0)
        test_dataset = test_dataset.filter(lambda x: x["HATEFUL"] > 0)

    elif 0 < negative_examples_proportion <= 1:

        def keep_example(example):
            return (example["HATEFUL"] > 0) or random.random() <= negative_examples_proportion

        train_dataset = train_dataset.filter(keep_example)
        # Don't filter dev and test

    else:
        print(f"{negative_examples_proportion} must be between 0 and 1")
        sys.exit(1)
    """
    print("Done")
    print(f"Train examples : {len(train_dataset):<5} (Hateful {sum(train_dataset['HATEFUL'])})")
    print(f"Dev examples   : {len(dev_dataset):<5}   (Hateful {sum(dev_dataset['HATEFUL'])})")
    print(f"Test examples  : {len(test_dataset):<5}  (Hateful {sum(test_dataset['HATEFUL'])})")


    trainer, test_dataset = train_finegrained(
        model, tokenizer, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
        context=context, batch_size=batch_size, eval_batch_size=eval_batch_size,
        accumulation_steps=accumulation_steps, epochs=epochs,
        use_class_weight=use_class_weight,
        warmup_ratio=warmup_ratio,
        output_dir=output_dir,
    )

    print("\n"*3, "Saving...")

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Models saved at {output_path}")

    print("Evaluation")

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

    test_results = eval_trainer.evaluate(test_dataset)
    for k, v in test_results.items():
        print(f"{k} = {v:.4f}")

if __name__ == '__main__':
    fire.Fire(train_finegrained_classifier)

