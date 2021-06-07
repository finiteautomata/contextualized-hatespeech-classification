"""
Script to train hatespeech classifier
"""
import os
import fire
from matplotlib import use
import torch
import sys
import random
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer, BertTokenizerFast
)
from hatedetection import BertForSequenceMultiClassification, load_datasets, extended_hate_categories
from hatedetection.training import tokenize, lengths
from hatedetection.preprocessing import special_tokens
from hatedetection.metrics import compute_category_metrics


def load_model_and_tokenizer(model_name, context, max_length=None, class_weight=None):
    """
    Load model and tokenizer
    """

    if not max_length:
        max_length = lengths[context]

    model = BertForSequenceMultiClassification.from_pretrained(
        model_name, return_dict=True, num_labels=len(extended_hate_categories),
        pos_weight=class_weight,
    )

    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_length

    vocab = tokenizer.get_vocab()
    new_tokens_to_add = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]

    if new_tokens_to_add:
        """
        TODO: Perdoname Wilkinson, te he fallado

        Hay una interfaz diferente acá, no entiendo bien por qué
        """
        if type(tokenizer) is BertTokenizerFast:
            tokenizer.add_special_tokens({'additional_special_tokens': new_tokens_to_add})
        else:
            tokenizer.add_special_tokens(new_tokens_to_add)
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer



def train_category_classifier(
    output_path, train_path=None, test_path=None, context='none',
    model_name = 'dccuchile/bert-base-spanish-wwm-cased', batch_size=32, eval_batch_size=16, output_dir=None,
    max_length=None, epochs=5, warmup_proportion=0.1, negative_examples_proportion=None, random_seed=2021, use_class_weight=False,
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
    print("Training hate speech category classifier")

    random.seed(random_seed)

    if context not in lengths.keys():
        print(f"{context} must be in {lengths.keys()}")
        sys.exit(1)

    if not output_dir:
        output_dir = os.path.join("results", output_path)

    print(f"Uses context: {context}")
    print(f"Tokenizer max length: {max_length}")
    print("*"*80, end="\n"*3)

    print("Loading datasets... ", end="")
    add_body = True if "body" in context else False

    train_dataset, dev_dataset, test_dataset = load_datasets(train_path, test_path, add_body=add_body)



    if negative_examples_proportion is None:
        """
        Only train and test on negative examples
        """
        train_dataset = train_dataset.filter(lambda x: x["HATEFUL"] > 0)
        dev_dataset = dev_dataset.filter(lambda x: x["HATEFUL"] > 0)
        test_dataset = test_dataset.filter(lambda x: x["HATEFUL"] > 0)

    elif 0 < negative_examples_proportion <= 1:

        def keep_example(example):
            return (example["HATEFUL"] > 0) or random.random() < negative_examples_proportion

        train_dataset = train_dataset.filter(keep_example)
        # dev_dataset = dev_dataset.filter(keep_example)
        # Don't filter test!

    else:
        print(f"{negative_examples_proportion} must be between 0 and 1")
        sys.exit(1)

    print("Done")
    print(f"Train examples : {len(train_dataset):<5} (Hateful {sum(train_dataset['HATEFUL'])})")
    print(f"Dev examples   : {len(dev_dataset):<5}   (Hateful {sum(dev_dataset['HATEFUL'])})")
    print(f"Test examples  : {len(test_dataset):<5}  (Hateful {sum(test_dataset['HATEFUL'])})")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = torch.Tensor([train_dataset[c] for c in extended_hate_categories]).T

    class_weight = (1 / (2 * labels.mean(0))).to(device) if use_class_weight else None


    print("")
    print("Loading model and tokenizer... ", end="")
    model, tokenizer = load_model_and_tokenizer(model_name, context, max_length, class_weight=class_weight)
    print("Done")


    my_tokenize = lambda batch: tokenize(tokenizer, batch, context=context)

    print("Tokenizing and formatting datasets...")
    train_dataset = train_dataset.map(my_tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(my_tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(my_tokenize, batched=True, batch_size=eval_batch_size)



    def format_dataset(dataset):
        def get_category_labels(examples):
            return {'labels': torch.Tensor([examples[cat] for cat in extended_hate_categories])}
        dataset = dataset.map(get_category_labels)
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)


    print("\n\n", "Sanity check")
    print(tokenizer.decode(train_dataset[0]["input_ids"]))

    """
    Finally, train!
    """

    print("\n"*3, "Training...")

    total_steps = (epochs * len(train_dataset)) // batch_size
    warmup_steps = int(warmup_proportion * total_steps)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="mean_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_category_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
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

