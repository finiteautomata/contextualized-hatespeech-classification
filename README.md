# Hatespeech Classification in News

Code for "Assessing the impact of contextual information in hate speech detection", Pérez, J. M., Luque, F., Zayat, D., Kondratzky, M., Moro, A., Serrati, P., Zajac, J., Miguel, P., Gravano, A. & Cotik, V. (2022). 

[Link to paper](https://arxiv.org/abs/2210.00465)

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
python bin/train_category_classifier.py --context 'none' --output_path models/bert-non-contextualized-hate-category-es/ --epochs 5

# Train contextualized model
python bin/train_hate_classifier.py --context 'title' --output_path models/bert-contextualized-hate-speech-es/ --epochs 10
python bin/train_category_classifier.py --context 'title' --output_path models/bert-contextualized-hate-category-es/ --epochs 5


# Train fully contextualized
# Check out notebooks/Hatespeech_Colab_TPU.ipynb
python bin/train_category_classifier.py --context 'title+body' --output_path models/bert-title+body-hate-speech-es/ --epochs 5 --batch_size 8 --eval_batch_size 8
python bin/train_hate_classifier.py --context 'title+body' --output_path models/bert-title+body-hate-speech-es/ --epochs 10
```

For more instructions, check [TRAIN_EVALUATE.md](TRAIN_EVALUATE.md)



## Finetuning

1. First, preprocess data

```bash
python bin/preprocess_finetune_data.py "/content/drive/Shareddrives/HateSpeech/data/hatespeech-data/" "/content/drive/MyDrive/data/finetune-news/finetune_data/" --num_workers 10
```

2. Run finetuning

```bash
python bin/xla_spawn.py --num_cores 8 bin/finetune_lm.py config/no_context_ft.json
```
