#!/bin/sh

# Analysis Type: ["dataset", "custom_message"]
analysis_type='dataset'

# Hyperparameters 
alpha=1.

# Data Path
train_data_path='./datasets/SMS_Spam_Collection_Dataset/preprocessed/train/data.pkl'
test_data_path='./datasets/SMS_Spam_Collection_Dataset/preprocessed/test/data.pkl'
parameters_path='./datasets/SMS_Spam_Collection_Dataset/preprocessed/train/parameters.pkl'

# Custom Message
custom_messages_path='./custom_messages.txt'

python ./main.py    --analysis-type $analysis_type \
                    --train-data-path $train_data_path \
                    --test-data-path $test_data_path \
                    --parameters-path $parameters_path \
                    --alpha $alpha \
                    --custom-messages-path $custom_messages_path \
                    --adversarial
