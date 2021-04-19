import os
import json
import pandas as pd
import pathlib
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, Value, ClassLabel, Features
from .categories import extended_hate_categories
from .preprocessing import preprocess_tweet


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")

_train_path = os.path.join(data_dir, "train.json")
_test_path = os.path.join(data_dir, "test.json")


def serialize(article, comment, add_body):
    """
    Serializes article and comment
    """
    ret = comment.copy()
    ret["title"] = article["title"]

    if add_body:
        ret["body"] = article["body"]
    return ret



def load_datasets(train_path=None, test_path=None, add_body=False, limit=None, preprocess=False):
    """
    Load and return datasets

    Returns
    -------

        train_dataset, dev_dataset, test_datasets: datasets.Dataset
    """
    test_path = test_path or _test_path
    train_path = train_path or _train_path

    with open(train_path) as f:
        train_articles = json.load(f)

    with open(test_path) as f:
        test_articles = json.load(f)


    train_comments = [serialize(article, comment, add_body) for article in train_articles for comment in article["comments"]]
    test_comments = [serialize(article, comment, add_body) for article in test_articles for comment in article["comments"]]

    if limit:
        train_comments = train_comments[:limit]
        test_comments = test_comments[:limit]
    train_df = pd.DataFrame(train_comments)
    test_df = pd.DataFrame(test_comments)

    train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=20212021)

    """
    Apply preprocessing: convert usernames to "usuario" and urls to URL
    """

    if preprocess:
        import pandarallel
        pandarallel.initialize()

        for df in [train_df, dev_df, test_df]:
            df["text"] = df["text"].parallel_apply(preprocess_tweet)
            df["title"] = df["title"].parallel_apply(preprocess_tweet)
            if add_body:
                """
                TODO: Perhaps tweet preprocessing is not suitable for the body
                """
                df["body"] = df["body"].parallel_apply(preprocess_tweet)

    features = Features({
        'id': Value('uint64'),
        'title': Value('string'),
        'text': Value('string'),
        'HATEFUL': ClassLabel(num_classes=2, names=["Not Hateful", "Hateful"])
    })

    if add_body:
        features["body"] = Value('string')


    for cat in extended_hate_categories:
        """
        Set for WOMEN, LGBTI...and also for CALLS
        """
        features[cat] = ClassLabel(num_classes=2, names=["NO", "YES"])

    columns = list(features.keys())

    train_dataset = Dataset.from_pandas(train_df[columns], features=features)
    dev_dataset = Dataset.from_pandas(dev_df[columns], features=features)
    test_dataset = Dataset.from_pandas(test_df[columns], features=features)

    return train_dataset, dev_dataset, test_dataset

