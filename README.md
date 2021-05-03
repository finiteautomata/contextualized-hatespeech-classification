# Hatespeech Classification in News

1. Get the dataset and put it under `data/`

2. Generate and preprocess dataset

```
python bin/create_dataset.py data/articles.json data/comments.json --train_path data/train.json --test_path data/test.json
python bin/preprocess_dataset.py
```


3. Train models

First, plain hate classifiers

```
# Train non-contextualized model
python bin/train_hate_classifier.py --context 'none' --output_path models/bert-non-contextualized-hate-speech-es/ --epochs 10
python bin/train_category_classifier.py --output_path models/bert-non-contextualized-hate-category-es/ --epochs 5

# Train contextualized model
python bin/train_hate_classifier.py --context 'title' --output_path models/bert-contextualized-hate-speech-es/ --epochs 10
python bin/train_category_classifier.py --use_context --output_path models/bert-contextualized-hate-category-es/ --epochs 5
```

python bin/train_hate_classifier.py --context 'title+body' --output_path models/bert-title+body-hate-speech-es/ --epochs 10
