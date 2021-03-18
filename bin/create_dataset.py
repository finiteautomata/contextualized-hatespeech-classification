import json
import fire
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from hatedetection import hate_categories


def process_comment(raw_comment):
    """
    Process raw comment
    """
    ret = {
        "text": raw_comment["text"],
        "HATEFUL": int(len(raw_comment['HATE']) >= 2)
    }
    for cat in hate_categories + ["CALLS"]:
        ret[cat] = 0

    if ret["HATEFUL"]:
        ret["CALLS"] = int(len(raw_comment['CALLS']) >= 2)

        for category in hate_categories:
            ret[category] = int(len(raw_comment[category]) > 0)

    return ret


def create_dataset(articles_path, comments_path, train_path, test_path, train_size=0.8, random_state=2021):
    """
    Creates train and test dataset
    """

    with open(articles_path) as f:
        raw_articles = json.load(f)

    with open(comments_path) as f:
        raw_comments = json.load(f)

    print(f"We have {len(raw_articles)} articles and {len(raw_comments)} comments")
    articles = {art['tweet_id']:art for art in raw_articles}

    for art in articles.values():
        art["comments"] = []

    counts = []


    for comment in tqdm(raw_comments):
        tweet_id = comment["article_id"]
        article = articles[tweet_id]
        article["comments"].append(
            process_comment(comment)
        )


    train_articles, test_articles = train_test_split(list(articles.values()),
        train_size=train_size, random_state=random_state
    )

    print(f"Train Articles: {len(train_articles)}")
    print(f"Test Articles: {len(test_articles)}")

    with open(train_path, "w") as f:
        json.dump(train_articles, f)

    with open(test_path, "w") as f:
        json.dump(test_articles, f)


if __name__ == '__main__':
    fire.Fire(create_dataset)