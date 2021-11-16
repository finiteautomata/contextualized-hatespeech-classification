# Evaluation

## Train


```bash
model_name="dccuchile/bert-base-spanish-wwm-cased"
python bin/train_finegrained.py --output_path models/test \
    --model_name $model_name \
    --batch_size 32 --eval_batch_size 32 --accumulation_steps 1 \
    --context 'none' \
    --use_class_weight \
    --epochs 10
```


### Run experiments
#### Task A

```bash
model_name="dccuchile/bert-base-spanish-wwm-cased"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context $context \
    --output_path "evaluations/beto_plain_${context}.json" \
    --plain \
    --epochs 5

context="text"
model_name="dccuchile/bert-base-spanish-wwm-cased"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context $context \
    --output_path "evaluations/beto_plain_${context}.json" \
    --plain \
    --batch_size 16 --accumulation_steps 2\
    --epochs 5

context="text+body"
model_name="dccuchile/bert-base-spanish-wwm-cased"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context 'title+body' \
    --output_path "evaluations/beto_plain_${context}.json" \
    --batch_size 8 --accumulation_steps 4 \
    --plain \
    --epochs 10

## Finetuned
context="none"
model_name="finiteautomata/betonews-nonecontext"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context $context \
    --output_path "evaluations/betonews_plain_${context}.json" \
    --plain \
    --epochs 5

context="text"
model_name="finiteautomata/betonews-tweetcontext"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context $context \
    --output_path "evaluations/betonews_plain_${context}.json" \
    --plain \
    --batch_size 16 --accumulation_steps 2\
    --epochs 5

context="text+body"
model_name="finiteautomata/betonews-bodycontext"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context $context \
    --plain \
    --output_path "evaluations/betonews_plain_${context}.json" \
    --batch_size 8 --accumulation_steps 4 \
    --epochs 10
```


#### Task B

```bash
model_name="dccuchile/bert-base-spanish-wwm-cased"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context 'none' \
    --output_path evaluations/beto_fine_none_weighted.json \
    --use_class_weight \
    --epochs 5

model_name="dccuchile/bert-base-spanish-wwm-cased"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context 'title' \
    --output_path evaluations/beto_fine_title_weighted.json \
    --use_class_weight \
    --epochs 10


model_name="dccuchile/bert-base-spanish-wwm-cased"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context 'title+body' \
    --output_path evaluations/beto_fine_titlebody.json \
    --batch_size 16 --eval_batch_size 16 --accumulation_steps 2 \
    --use_class_weight \
    --epochs 10
```

### Robertuito

```bash
model_name="finiteautomata/robertuito-base-cased"
context="text"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context $context \
    --output_path "evaluations/robertuito_cased_fine_${context}.json" \
    --max_length 128 \
    --batch_size 32 \
    --use_class_weight \
    --epochs 10

model_name="finiteautomata/robertuito-base-uncased"
context="text"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context $context \
    --output_path "evaluations/robertuito_uncased_fine_${context}.json" \
    --max_length 128 \
    --batch_size 32 \
    --use_class_weight \
    --epochs 10

#
#
# Finetuned
#
#


model_name="finiteautomata/robertuitonews-tweetcontext"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context 'text' \
    --output_path "evaluations/robertuitonews_fine_${context}.json" \
    --max_length 128 \
    --batch_size 32  \
    --use_class_weight \
    --epochs 10


model_name="finiteautomata/robertuitonews-cased-tweetcontext"
python bin/run_experiments.py --model_name $model_name --times 10 \
    --context 'text' \
    --output_path "evaluations/robertuitonews_cased_fine_${context}.json" \
    --max_length 128 \
    --batch_size 32  \
    --use_class_weight \
    --epochs 10
```



## Single

###
### No context
###
for i in {1..15}
do
    model_path="models/bert-single-no-context-weight_${i}/"
    context="none"
    if [ -d "$model_path" ]; then
      echo "$model_path exists. -- continue"
      continue
    fi
    python bin/train_category_classifier.py --context $context --output_path $model_path --epochs 10 --negative_examples_proportion 1.0 --use_class_weight
done

###
### Title
###

for i in {1..15}
do
    model_path="models/bert-single-title-weight_${i}/"
    context="title"
    if [ -d "$model_path" ]; then
      echo "$model_path exists. -- continue"
      continue
    fi
    python bin/train_category_classifier.py --context $context --output_path $model_path --epochs 10 --negative_examples_proportion 1.0 --use_class_weight
done


for i in {1..15}
do
    model_path="models/bert-ft10k-single-title-${i}/"
    model_name="models/beto-cased-news-10k"
    context="title"
    if [ -d "$model_path" ]; then
      echo "$model_path exists. -- continue"
      continue
    fi
    python bin/train_category_classifier.py \
        --context $context \
        --model_name $model_name \
        --output_path $model_path \
        --epochs 10 \
        --negative_examples_proportion 1.0 \
        --use_class_weight
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

for model_path in models/*-single-*
do
    base=`basename "$model_path"`
    output_path="evaluations/${base#bert-}.json"
    if test -f "$output_path"; then
      echo "$output_path exists. -- continue"
      continue
    fi
    context="none"
    if [[ "$model_path" == *"title-body"* ]]; then
        context="title-body"
    elif [[ "$model_path" == *"title"* ]]; then
        context="title"
    elif [[ "$model_path" == *"no-context"* ]]; then
        context="none"
    fi

    echo "=============================================="
    echo "Context = ${context}"
    python bin/eval_hate_category.py --context "$context" --model_name $model_path --output_path $output_path --use_all
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

