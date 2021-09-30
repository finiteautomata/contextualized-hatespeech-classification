import logging
import fire
import json
import glob
import os
from hatedetection.preprocessing import preprocess
from multiprocessing import Pool
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig()

logger = logging.getLogger('preprocess_finetune_data')
logger.setLevel(logging.INFO)

def preprocess_article(article):
    article["text"] = preprocess(article["text"])

    for comment in article["comments"]:
        comment["text"] = preprocess(comment["text"])
    return article

selected_media = [
    "infobae",
    "clarincom",
    "LANACION",
    "cronica",
    "pagina12",
    "perfilcom",
    "izquierdadiario",
    "laderechadiario",
    "laderechamedios",
]

user_mapping = {
    "laderechamedios": "laderechadiario",
}

def preprocess_finetune_data(data_path:str, output_path:str, num_workers:int=10, filter_media:bool=True, min_body_len:int=500, num_chunks:int=100):
    """
    Preprocess data for finetuning BERT models

    Arguments:
    ---------

    data_path: path
        Directory where to look for json containing articles

    output_path:
        Where to save the json of preprocessed articles

    num_workers: int

    """
    logger.info(f"Opening {data_path}")

    files = glob.glob(os.path.join(data_path, "*.json"))

    logger.info(f"Found {len(files)}")
    articles = []

    news_count = {}

    for file in tqdm(files):
        with open(file) as f:
            news = json.load(f)

            for article in news:
                if filter_media and article["user"] not in selected_media:
                    continue

                article["body"] = article.get("body", "").strip()

                if len(article["body"]) < min_body_len:
                    continue
                """
                Ok, now we can add
                """
                article["user"] = user_mapping.get(article["user"], article["user"])
                articles.append(article)

                news_count[article["user"]] = news_count.get(article["user"], 0) + 1


    logger.info(f"Loaded {len(articles)} articles")
    logger.info(news_count)
    """

    """
    logger.info(f"Creating {num_workers} workers")

    pool = Pool(num_workers)
    pbar = tqdm(total=len(articles))

    preprocessed_articles = []

    for res in pool.imap(preprocess_article, articles):
        preprocessed_articles.append(res)
        pbar.update()

    logger.info("Saving...")

    comments = []

    for art in preprocessed_articles:
        for comment in art["comments"]:
            comments.append(
                {
                    "article_text": art["text"],
                    "article_title": art["title"],
                    "body": art["body"],
                    "text": comment["text"]
                }
            )

    """
    Save in chunks

    """

    n = len(comments) // num_chunks
    for i, chunk in enumerate(comments[i:i + n] for i in range(0, len(comments), n)):
        file_path = os.path.join(output_path, f"{i+1:03}.json")

        with open(file_path, "w+") as f:
            json.dump(
                {"data":chunk}, f, indent=4)
            logger.info(f"Saved {len(chunk)} comments to {file_path}")


if __name__ == '__main__':
    fire.Fire(preprocess_finetune_data)

