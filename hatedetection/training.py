from torch.nn import BCEWithLogitsLoss
from .metrics import compute_hate_metrics
from .preprocessing import special_tokens
import torch
from transformers import (
    BertTokenizerFast, TrainingArguments, Trainer,
    AutoModelForSequenceClassification, AutoTokenizer
)
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
from .categories import extended_hate_categories
from .metrics import compute_extended_category_metrics


lengths = {
    'none': 128,
    'title-only': 128,
    'title': 256,
    'title-hyphen': 256,
    'body': 512,
    'title+body': 512,
}


def load_tokenizer(model_name, max_length, model=None, tokenizer_class=AutoTokenizer):
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        never_split=special_tokens
    )
    vocab = tokenizer.get_vocab()
    new_tokens_to_add = [tok for tok in special_tokens if tok not in vocab]

    if new_tokens_to_add:
        tokenizer.add_tokens(new_tokens_to_add)
        if model:
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.model_max_length = max_length
    return tokenizer

def load_model_and_tokenizer(model_name, num_labels, device, add_tokens=special_tokens, max_length=128):
    """
    Load model and tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, return_dict=True, num_labels=num_labels
    )


    model = model.to(device)
    model.train()
    tokenizer = load_tokenizer(model_name, max_length, model=model)

    return model, tokenizer


class MultiLabelTrainer(Trainer):
    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_hatespeech_model_and_tokenizer(model_name, context, max_length=None):
    """
    Load model and tokenizer for hate speech classification
    """

    if not max_length:
        max_length = lengths[context]

    id2label = {0: 'Not hateful', 1: 'Hateful'}
    label2id = {v:k for k,v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, return_dict=True, num_labels=2
    )

    model.config.id2label = id2label
    model.config.label2id = label2id

    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_length

    """
    Check for new tokens
    """

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


def tokenize(tokenizer, batch, context, padding='max_length', truncation='longest_first', context_first=False):
    """
    Apply tokenization

    Arguments:
    ---------

    context: string
        Type of allowed context. Options are ['none', 'title', 'title-only', 'body', 'title+body']
    """

    valid_contexts = {'none', 'title', 'title-only', 'body', 'title-hyphen', 'title+body'}

    if context not in valid_contexts:
        raise ValueError(f"Invalid context. Must be one of {valid_contexts}")

    kwargs = { 'padding': padding, 'truncation': truncation }

    if context == "title-hyphen":
        """
        Special case, we have to merge
        """
        hyphened_input = [text+" - " + title for text, title in zip(batch["text"], batch["title"])]
        return tokenizer(hyphened_input, **kwargs)

    if context == 'title':
        tokenize_args = [
            batch['text'],
            batch['title']
        ]
    elif context == 'body':
        tokenize_args = [
            batch['text'],
            batch['body']
        ]
    elif context == "title+body":
        tokenize_args = [
            batch['text'],
            [title + " - "+ body for title, body in zip(batch['title'],batch['body'])],
        ]
    elif context == 'title-only':
        tokenize_args = [
            batch['title']
        ]
    elif context == 'none':
        tokenize_args = [
            batch['text']
        ]

    if context_first:
        # Invert it
        tokenize_args = tokenize_args[::-1]

    return tokenizer(*tokenize_args, **kwargs)


def train_hatespeech_classifier(
    model, train_dataset, dev_dataset,
    batch_size, eval_batch_size, output_dir, epochs=10, warmup_proportion=0.1,
    load_best_model_at_end=True, metric_for_best_model="f1", **kwargs):
    """
    Train hate speech classifier
    """

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
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        **kwargs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_hate_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()

    return trainer, dev_dataset


def train_finegrained(
    model, tokenizer, train_dataset, dev_dataset, test_dataset, context,
    batch_size=32, eval_batch_size=32,
    accumulation_steps=1, epochs=5, warmup_ratio=0.1,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    print(
        sorted(
            set(len(x) for x in train_dataset["input_ids"])
        )
    )

    """
    Finally, train!
    """

    print("\n"*3, "Training...")


    output_path = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        gradient_accumulation_steps=accumulation_steps,
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

    return trainer, test_dataset