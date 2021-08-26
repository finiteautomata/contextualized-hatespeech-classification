from torch.nn import BCEWithLogitsLoss
from .metrics import compute_hate_metrics
from .preprocessing import special_tokens
from transformers import (
    BertTokenizerFast, TrainingArguments, Trainer,
    AutoModelForSequenceClassification, AutoTokenizer
)

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
