"""
Script to train hatespeech classifier
"""
import os
import fire
import tempfile
from matplotlib import use
import tempfile
import torch
import sys
import random
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer, BertTokenizerFast, DataCollatorWithPadding
)
from hatedetection import BertForSequenceMultiClassification, load_datasets, extended_hate_categories
from hatedetection.training import tokenize, lengths, load_model_and_tokenizer, MultiLabelTrainer, lengths
from hatedetection.metrics import compute_extended_category_metrics



def train_category_classifier(
    output_path, train_path=None, test_path=None, context='none',
    model_name = 'dccuchile/bert-base-spanish-wwm-cased', batch_size=16, eval_batch_size=16, output_dir=None,
    accumulation_steps=2, max_length=None, epochs=5, warmup_ratio=0.1, negative_examples_proportion=1,
    random_seed=2021, use_class_weight=False, use_dynamic_padding=True,
    ):

    """
    Train and save hatespeech classifier

    Arguments:
    ----------

    train_path:
        Path to training data
    test_path:
        Path to test data
    """
    print("*"*80)
    print(f"Training hate speech category classifier -- {output_path}")

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
    print(f"Negative examples: {negative_examples_proportion}")
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


    labels = torch.Tensor([train_dataset[c] for c in extended_hate_categories]).T

    class_weight = (1 / (2 * labels.mean(0))).to(device) if use_class_weight else None

    print(f"Class weight: {class_weight}")

    print("")
    print("Loading model and tokenizer... ", end="")
    print("Done")

    padding = False if use_dynamic_padding else 'max_length'

    my_tokenize = lambda batch: tokenize(tokenizer, batch, context=context, padding=padding)

    print("Tokenizing and formatting datasets...")
    train_dataset = train_dataset.map(my_tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(my_tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(my_tokenize, batched=True, batch_size=eval_batch_size)

    def format_dataset(dataset):
        def get_category_labels(examples):
            return {'labels': torch.Tensor([examples[cat] for cat in extended_hate_categories])}
        dataset = dataset.map(get_category_labels)

        if use_dynamic_padding:
            return dataset

        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset

    data_collator = DataCollatorWithPadding(tokenizer, padding="longest") if use_dynamic_padding else None

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)


    print("\n\n", "Sanity check")
    print(tokenizer.decode(train_dataset[0]["input_ids"]))

    """
    Finally, train!
    """

    print("\n"*3, "Training...")


    output_path = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        gradient_accumulation_steps = accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="mean_f1",
        group_by_length=True,
    )


    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        class_weight=class_weight,
        compute_metrics=lambda pred: compute_extended_category_metrics(dev_dataset, pred),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    """
    Evaluate
    """
    trainer.evaluate(dev_dataset)

    print("\n"*3, "Saving...")

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Models saved at {output_path}")

if __name__ == '__main__':
    fire.Fire(train_category_classifier)

