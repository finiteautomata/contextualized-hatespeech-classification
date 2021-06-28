import fire
import os
import torch
import ijson
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from hatedetection.training import tokenize
from hatedetection.preprocessing import preprocess_tweet
from torch.utils.data import DataLoader
from datasets import Dataset, Value, ClassLabel, Features
from hatedetection import extended_hate_categories
from pandarallel import pandarallel
from hatedetection import HateSpeechAnalyzer

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_outputs(model, dataloader):
    outputs = []

    for batch in tqdm(dataloader):
        batch = {k:v.to(device) for k, v in batch.items() if k != "id"}
        outs = model(**batch)
        outputs.append((outs.logits > 0).cpu().numpy())

    return np.vstack(outputs)



def build_dataset(df, tokenizer, batch_size):
    features = Features({
        'id': Value('uint64'),
        'context': Value('string'),
        'text': Value('string'),
    })


    dataset = Dataset.from_pandas(df, features=features)

    dataset = dataset.map(
        lambda x: tokenizer(x["text"], x["context"], padding="longest", truncation='longest_first'),
        batched=True, batch_size=batch_size,
    )
    def format_dataset(dataset):
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
        return dataset

    dataset = format_dataset(dataset)

    return dataset

def process_comments(model, tokenizer, data, batch_size):
    df = pd.DataFrame(data)

    df["text"] = df["text"].parallel_apply(preprocess_tweet)
    df["context"] = df["context"].parallel_apply(preprocess_tweet)
    batch_size = 16

    dataset = build_dataset(df, tokenizer, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    outputs = get_outputs(model, dataloader)


    df[extended_hate_categories] = outputs
    df["HATEFUL"] = df[extended_hate_categories[1:]].sum(axis=1) > 0

    return df



def predict_unlabeled_data(path, output_dir, eval_batch_size=16, comments_batch_size=80_000):
    """
    Predicts unlabelled data

    Arguments:
    ----------

    path: string
        Path to JSON with articles and comments
    """
    print("Loading contextualized model...")
    analyzer = HateSpeechAnalyzer.load_contextualized_model()


    model = analyzer.base_model
    model = model.to(device)
    model.eval()

    tokenizer = analyzer.tokenizer

    pandarallel.initialize()

    # TODO: magic number
    num_articles = 537_200

    data = []

    num_comments = 0
    batch_num = 1

    with open(path) as f:
        articles = ijson.items(f, 'item')
        for article in tqdm(articles, total=num_articles):
            num_comments += len(article["comments"])

            context = article["title"] if "title" in article else article["text"]

            for comment in article["comments"]:
                data.append({
                    "user": article["user"],
                    "text": comment["text"],
                    "context": context,
                    "id": comment["tweet_id"],
                })

            if len(data) > comments_batch_size:
                output_path = os.path.join(output_dir, f"{batch_num}.csv")
                df = process_comments(model, tokenizer, data, batch_size=eval_batch_size)

                df.to_csv(output_path, columns=["id"] + extended_hate_categories)
                data = []
                batch_num += 1

    print(num_comments, " comments processed")




if __name__ == '__main__':
    fire.Fire(predict_unlabeled_data)