"""
Script to train hatespeech classifier
"""
import sys
import fire
import torch
import transformers
from transformers import (
    Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer,
    BertTokenizerFast
)
from hatedetection import load_datasets
from hatedetection.metrics import compute_hate_metrics
from hatedetection.preprocessing import special_tokens

def load_model_and_tokenizer(model_name, max_length):
    """
    Load model and tokenizer
    """
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

def tokenize(tokenizer, batch, context, padding='max_length', truncation='longest_first'):
    """
    Apply tokenization

    Arguments:
    ---------

    use_context: boolean (default True)
        Whether to add the context to the
    """

    if context == 'title':
        args = [batch['context'], batch['text']]
    elif context == 'body':
        args = [batch['body'], batch['text']]
    else:
        args = [batch['text']]

    return tokenizer(*args, padding='max_length', truncation=True)



def train_hatespeech_classifier(
    output_path, train_path=None, test_path=None, context='none',
    model_name = 'dccuchile/bert-base-spanish-wwm-uncased', batch_size=32, eval_batch_size=16,
    max_length=None, epochs=10, warmup_proportion=0.1,
    ):

    """
    Train and save hatespeech classifier

    Arguments:
    ----------
    output_path:
        Where we save the classifier
    train_path:
        Path to training data
    test_path:
        Path to test data

    context: string
        One of {'none', 'title', 'body'}
    """
    print("*"*80)
    print("Training hate speech classifier")

    allowed_contexts = {'none', 'title', 'body'}
    if context not in allowed_contexts:
        print("")
        sys.exit(1)

    print(f"Uses context: {context}")

    lengths = {
        'none': 128,
        'title': 256,
        'body': 512
    }
    if not max_length:
        max_length = lengths[context]
    print(f"Tokenizer max length: {max_length}")
    print("*"*80, end="\n"*3)

    print("Loading datasets... ", end="")
    add_body = True if context == 'body' else False
    train_dataset, dev_dataset, test_dataset = load_datasets(train_path, test_path, add_body=add_body)

    print("Done")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("")
    print("Loading model and tokenizer... ", end="")
    model, tokenizer = load_model_and_tokenizer(model_name, max_length)
    print("Done")


    my_tokenize = lambda batch: tokenize(tokenizer, batch, context=context)

    print("Tokenizing and formatting datasets...")
    train_dataset = train_dataset.map(my_tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(my_tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(my_tokenize, batched=True, batch_size=eval_batch_size)


    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['HATEFUL']})
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)

    """
    Finally, train!
    """

    total_steps = (epochs * len(train_dataset)) // batch_size
    warmup_steps = int(warmup_proportion * total_steps)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    results = []

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_hate_metrics,
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
    fire.Fire(train_hatespeech_classifier)

