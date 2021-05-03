"""
Script to train hatespeech classifier
"""
import sys
import fire
import torch
import transformers
from hatedetection import load_datasets
from hatedetection.training import (
    train_hatespeech_classifier, tokenize, load_hatespeech_model_and_tokenizer
)


def train_model(
    output_path, train_path=None, test_path=None, context='none',
    model_name = 'dccuchile/bert-base-spanish-wwm-uncased', batch_size=32, eval_batch_size=16,
    max_length=None, epochs=10, warmup_proportion=0.1,
    ):
    """
    """

    print("*"*80)
    print("Training hate speech classifier")

    allowed_contexts = {'none', 'title', 'body', 'title+body'}

    if context not in allowed_contexts:
        print("")
        sys.exit(1)

    print(f"Uses context: {context}")
    print(f"Tokenizer max length: {max_length}")
    print("*"*80, end="\n"*3)

    print("Loading datasets... ", end="")

    add_body = True if "body" in context else False
    train_dataset, dev_dataset, test_dataset = load_datasets(train_path, test_path, add_body=add_body)

    print("Done")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("")
    print("Loading model and tokenizer... ", end="")
    model, tokenizer = load_hatespeech_model_and_tokenizer(model_name, context, max_length=max_length)
    print("Done")


    print("Tokenizing and formatting datasets...")
    my_tokenize = lambda batch: tokenize(tokenizer, batch, context=context)
    train_dataset = train_dataset.map(my_tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(my_tokenize, batched=True, batch_size=eval_batch_size)


    def format_dataset(dataset):
        dataset = dataset.map(lambda examples:{'labels': examples['HATEFUL']})
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)

    """
    Finally, train!
    """


    trainer, dev_dataset = train_hatespeech_classifier(
        model, train_dataset=train_dataset, dev_dataset=dev_dataset,
        batch_size=batch_size, eval_batch_size=eval_batch_size,
        epochs=epochs, warmup_proportion=warmup_proportion,
    )

    evaluation = trainer.evaluate(dev_dataset)

    print("\n*3", f"Evaluation: {evaluation}")


    print("\n"*3, "Saving...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Models saved at {output_path}")

if __name__ == '__main__':
    fire.Fire(train_model)

