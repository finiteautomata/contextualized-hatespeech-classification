# Evaluation

## Train

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

    python bin/train_hate_classifier.py --context $context --output_path $output_path --epochs 10 --output_dir $results_dir
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

for i in {1..15}
do
    model_path="models/bert-title-only-hate-speech-es_${i}/"
    output_dir="./results-title-only/${i}"
    echo $output_dir
    python bin/train_hate_classifier.py --context 'title-only' --output_path $model_path --epochs 5 --output_dir $output_dir
    rm -Rf $output_dir
done

## Single


for i in {4..9}
do
    model_path="models/bert-single-title_${i}/"
    output_dir="./results_single_no_weight/${i}"
    echo $output_dir
    python bin/train_category_classifier.py --context title --output_path $model_path --epochs 10 --negative_examples_proportion 1.0
    rm -Rf $output_dir
done

for i in {5..10}
do
    model_path="models/bert-single-title-weight_${i}/"
    output_dir="./results_single_weight/${i}"
    echo $output_dir
    python bin/train_category_classifier.py --context title --output_path $model_path --epochs 10 --negative_examples_proportion 1.0 --use_class_weight
    rm -Rf $output_dir
done



for i in {4..9}
do
    model_path="models/bert-single-title-es_${i}/"
    output_dir="./results_contextualized/${i}"
    echo $output_dir
    python bin/train_category_classifier.py --context 'title' --output_path $model_path --epochs 5 --output_dir $output_dir --negative_examples_proportion 0.2
    rm -Rf $output_dir
done


for i in {1..10}
do
    model_path="models/bert-single-title-body-hate-category-es_${i}/"
    output_dir="./results_contextualized/${i}"
    echo $output_dir
    python bin/train_category_classifier.py --context 'title+body' --output_path $model_path --epochs 5 --output_dir $output_dir --add_negative_examples_proportion 0.2 --batch_size 8 --eval_batch_size 8
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

for i in {1..15}
do
    model_path="models/bert-title-only-hate-category-es_${i}/"
    output_dir="./results-title-only/${i}"
    echo $output_dir
    python bin/train_category_classifier.py --context 'title-only' --output_path $model_path --epochs 5 --output_dir $output_dir
    rm -Rf $output_dir
done
```

### Evaluation

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
    python bin/eval_hate_speech.py --context 'none' --model_name $model_path --output_path $output_dir
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
    python bin/eval_hate_speech.py --context 'title' --model_name $model_path --output_path $output_dir
done

for i in {1..15}
do
    model_path="models/bert-hyphen-hate-speech-es_${i}/"
    output_path="evaluations/title-hyphen_${i}.json"
    echo $model_path
    echo $output_path
    python bin/eval_hate_speech.py --context 'title-hyphen' --model_name $model_path --output_path $output_path
done

for i in {1..15}
do
    model_path="models/bert-title-only-hate-speech-es_${i}/"
    output_path="evaluations/title-only_${i}.json"
    echo $model_path
    echo $output_path
    python bin/eval_hate_speech.py --context 'title-only' --model_name $model_path --output_path $output_path
done

## Single

for model_path in models/*-single-title*
do
    base=`basename "$model_path"`
    output_path="evaluations/dev/${base#bert-}.json"
    context="none"
    if [[ "$model_path" == *"title-body"* ]]; then
        context="title-body"
    elif [[ "$model_path" == *"title"* ]]; then
        context="title"
    fi
    echo "Context = ${context}"
    python bin/eval_hate_category.py --context "$context" --model_name $model_path --output_path $output_path --use_all --dev
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

for i in {1..15}
do
    model_path="models/bert-title-only-hate-category-es_${i}/"
    output_dir="./evaluations/title-only-category-${i}.json"
    echo $model_path
    echo $output_dir
    python bin/eval_hate_category.py --context 'title-only' --model_name $model_path --output_path $output_dir
done
```

