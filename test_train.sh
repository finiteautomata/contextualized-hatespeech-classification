TEST_OUTPUT_PATH=models/test

python bin/train_hate_classifier.py --train_path data/train.small.json \
    --output_path $TEST_OUTPUT_PATH --epochs 3

rm -Rf $TEST_OUTPUT_PATH && rm -Rf results/