import os
import json
import fire
import pathlib
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from hatedetection.preprocessing import preprocess_tweet
import multiprocessing as mp


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")


_train_path = os.path.join(data_dir, "train.json")
_test_path = os.path.join(data_dir, "test.json")




def preprocess_datasets(train_path=None, test_path=None):
    """
    Preprocess datasets
    """
    test_path = test_path or _test_path
    train_path = train_path or _train_path

    with open(train_path) as f:
        train_articles = json.load(f)

    with open(test_path) as f:
        test_articles = json.load(f)

    pool = mp.Pool(processes=None)

    for articles in [train_articles, test_articles]:
        for article in tqdm(articles):
            comments = [t["original_text"] for t in article["comments"]]

            full_texts = [article["title"], article["body"], article["tweet_text"]] + comments
            processed_text = pool.map(preprocess_tweet, full_texts)

            article["title"] = processed_text[0]
            article["body"] = processed_text[1]
            article["tweet"] = processed_text[2]

            for comm, val in zip(article["comments"], processed_text[3:]):
                comm["text"] = val

    """
    Saving...
    """
    print("Saving")
    with open(train_path, "w") as f:
        json.dump(train_articles, f, indent=4)

    with open(test_path, "w") as f:
        json.dump(test_articles, f, indent=4)






if __name__ == '__main__':
    fire.Fire(preprocess_datasets)