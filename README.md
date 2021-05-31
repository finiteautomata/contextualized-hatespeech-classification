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
python bin/train_category_classifier.py --context 'none' --output_path models/bert-non-contextualized-hate-category-es/ --epochs 5

# Train contextualized model
python bin/train_hate_classifier.py --context 'title' --output_path models/bert-contextualized-hate-speech-es/ --epochs 10
python bin/train_category_classifier.py --context 'title' --output_path models/bert-contextualized-hate-category-es/ --epochs 5


# Train fully contextualized
# Check out notebooks/Hatespeech_Colab_TPU.ipynb
python bin/train_category_classifier.py --context 'title+body' --output_path models/bert-title+body-hate-speech-es/ --epochs 5 --batch_size 8 --eval_batch_size 8
python bin/train_hate_classifier.py --context 'title+body' --output_path models/bert-title+body-hate-speech-es/ --epochs 10
```

### Multiple experiments


Train multiple models
```bash
for i in {1..15}
do
    echo "models/bert-contextualized-hate-speech-es_${i}/"
    output_dir="./results_contextualized/${i}"
    echo $output_dir
    CUDA_VISIBLE_DEVICES=1 python bin/train_hate_classifier.py --context 'title' --output_path "models/bert-contextualized-hate-speech-es_${i}/" --epochs 10 --output_dir $output_dir
    rm -Rf $output_dir
done

for i in {1..15}
do
    output_path="models/bert-hyphen-hate-speech-es_${i}/"
    echo $output_path
    results_dir="./results_hyphen/${i}"
    echo $results_dir
    context="title-hyphen"

    CUDA_VISIBLE_DEVICES=1 python bin/train_hate_classifier.py --context $context --output_path $output_path --epochs 10 --output_dir $results_dir
    rm -Rf $results_dir
done


for i in {1..15}
do
    model_path="models/bert-non-contextualized-hate-speech-es_${i}/"
    output_dir="./results_non_contextualized/${i}"
    echo $output_dir
    CUDA_VISIBLE_DEVICES=1 python bin/train_hate_classifier.py --context 'none' --output_path $model_path --epochs 10 --output_dir $output_dir
    rm -Rf $output_dir
done



## Category

for i in {1..15}
do
    model_path="models/bert-non-contextualized-hate-category-es_${i}/"
    output_dir="./results_non_contextualized/${i}"
    echo $output_dir
    python bin/train_category_classifier.py --context 'none' --output_path $model_path --epochs 5 --output_dir $output_dir
    rm -Rf $output_dir
done


for i in {1..15}
do
    model_path="models/bert-contextualized-hate-category-es_${i}/"
    output_dir="./results_contextualized/${i}"
    echo $output_dir
    CUDA_VISIBLE_DEVICES=1 python bin/train_category_classifier.py --context 'title' --output_path $model_path --epochs 5 --output_dir $output_dir
    rm -Rf $output_dir
done

for i in {1..15}
do
    model_path="models/bert-title-body-hate-category-es_${i}/"
    output_dir="./results_contextualized/${i}"
    echo $output_dir
    CUDA_VISIBLE_DEVICES=1 python bin/train_category_classifier.py --context 'title+body' --output_path $model_path --epochs 5 --output_dir $output_dir --batch_size 8 --eval_batch_size 8
    rm -Rf $output_dir
done
```

### Eval multiple times

```bash

for i in {1..15}
do
    if [[ $i -eq 1 ]];
    then
        model_path="models/bert-non-contextualized-hate-speech-es/"
    else
        model_path="models/bert-non-contextualized-hate-speech-es_${i}/"
    fi
    output_dir="./evaluations/non-context-${i}.json"
    echo $model_path
    echo $output_dir
    CUDA_VISIBLE_DEVICES=1 python bin/eval_hate_speech.py --context 'none' --model_name $model_path --output_path $output_dir
done


for i in {1..15}
do
    if [[ $i -eq 1 ]];
    then
        model_path="models/bert-contextualized-hate-speech-es/"
    else
        model_path="models/bert-contextualized-hate-speech-es_${i}/"
    fi
    output_dir="./evaluations/context-${i}.json"
    echo $model_path
    echo $output_dir
    CUDA_VISIBLE_DEVICES=1 python bin/eval_hate_speech.py --context 'title' --model_name $model_path --output_path $output_dir
done

## Category

for i in {0..15}
do
    if [[ $i -eq 0 ]];
    then
        model_path="models/bert-non-contextualized-hate-category-es/"
    else
        model_path="models/bert-non-contextualized-hate-category-es_${i}/"
    fi
    output_dir="./evaluations/non-context-category-${i}.json"
    echo $model_path
    echo $output_dir
    python bin/eval_hate_category.py --context 'none' --model_name $model_path --output_path $output_dir
done


for i in {0..15}
do
    if [[ $i -eq 0 ]];
    then
        model_path="models/bert-contextualized-hate-category-es/"
    else
        model_path="models/bert-contextualized-hate-category-es_${i}/"
    fi
    output_dir="./evaluations/context-category-${i}.json"
    echo $model_path
    echo $output_dir
    python bin/eval_hate_category.py --context 'title' --model_name $model_path --output_path $output_dir
done

for i in {1..15}
do
    model_path="models/bert-title-body-hate-category-es_${i}/"
    output_dir="./evaluations/title-body-category-${i}.json"
    echo $model_path
    echo $output_dir
    python bin/eval_hate_category.py --context 'title+body' --model_name $model_path --output_path $output_dir
done
```
