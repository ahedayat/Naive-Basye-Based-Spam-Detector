import os
import string
import pandas as pd
import numpy as np


def _main2():
    df = pd.read_csv('./raw/spam.csv', encoding='ISO-8859-1')
    df['sms_message'] = df['v2']
    df['spam'] = np.where(df['v1'] == 'spam', 1, 0)
    df = df[['sms_message', 'spam']]
    print(len(df))
    sample_df = df.sample(frac=0.25)
    print(len(sample_df))
    spam_df = sample_df.loc[df['spam'] == 1]
    ham_df = sample_df.loc[df['spam'] == 0]
    print(len(spam_df))
    print(len(ham_df))


def cleansing(df):
    df['sms_message'] = df['v2']
    # df['spam'] = np.where(df['v1'] == 'spam', 1, 0)
    # df = df[['sms_message', 'spam']]
    df['label'] = df['v1']
    df = df[['sms_message', 'label']]

    df['sms_message'] = df['sms_message'].str.replace(
        '\W+', ' ').str.replace('\s+', ' ').str.strip()
    df['sms_message'] = df['sms_message'].str.lower()
    df['sms_message'] = df['sms_message'].str.split()

    return df


def save_df(saving_path, saving_file_name, data):

    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    data.to_pickle('{}/{}.pkl'.format(saving_path, saving_file_name))


def get_vocabulary(df):
    vocabulary = list(set(df['sms_message'].sum()))

    letters = dict()
    for word in vocabulary:
        for word_letter in word:
            if word_letter not in letters.keys():
                letters[word_letter] = 0
            letters[word_letter] += 1
    letters = pd.DataFrame(letters.items(), columns=['letter', 'num'])
    letters = letters.sort_values(by='num', ascending=False)

    word_counts_per_sms = list()
    for ix, (_, row) in enumerate(df.iterrows()):
        counts = list()
        for word in vocabulary:
            counts.append(row[0].count(word))
        word_counts_per_sms.append(counts)

        progress_percent = ((ix+1)/len(df))*100
        print("%d/%d (%.2f%%)" % (ix+1, len(df), progress_percent), end='\r')

    print()

    word_counts_per_sms = pd.DataFrame(word_counts_per_sms, columns=vocabulary)

    df = pd.concat(
        [df.reset_index(), word_counts_per_sms], axis=1).iloc[:, 1:]

    return vocabulary, letters, word_counts_per_sms, df


def preprocess(file_path, train_data_percent, saving_path):
    # Read Data
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Cleansing Data
    df = cleansing(df)

    # Sample Training Data
    train_data = df.sample(
        frac=train_data_percent, random_state=1).reset_index(drop=True)

    # Sample Test Data
    test_data = df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    # Prepare the vocabulary and count the number of separate words in each message
    vocabulary, letters, word_counts_per_sms, train_data = get_vocabulary(
        df=train_data)

    # Caculating Parameters
    Pspam = train_data['label'].value_counts()['spam'] / train_data.shape[0]
    Nspam = train_data.loc[train_data['label']
                           == 'spam', 'sms_message'].apply(len).sum()

    Pham = train_data['label'].value_counts()['ham'] / train_data.shape[0]
    Nham = train_data.loc[train_data['label'] == 'ham',
                          'sms_message'].apply(len).sum()

    Nvoc = len(train_data.columns) - 3

    # Preparing Parameters Data Frame
    parameters = {'pspam': Pspam,
                  'nspam': Nspam,
                  'pham': Pham,
                  'nham': Nham,
                  'nvoc': Nvoc}

    parameters = pd.DataFrame(parameters.items(), columns=['par', 'value'])

    # Saving Train Data Frames
    save_df(saving_path='{}/train'.format(saving_path),
            saving_file_name='data', data=train_data)
    save_df(saving_path='{}/train'.format(saving_path),
            saving_file_name='parameters', data=parameters)

    # Saving Test Data Frames
    save_df(saving_path='{}/test'.format(saving_path),
            saving_file_name='data', data=test_data)
    save_df(saving_path='{}/test'.format(saving_path),
            saving_file_name='parameters', data=parameters)

    # Saving Letters Repeatation
    save_df(saving_path='{}'.format(saving_path),
            saving_file_name='letters', data=letters)


def _main():
    # Raw Data Parameters
    file_path = './raw/spam.csv'

    # Preprocessed Data Parameters
    saving_path = './preprocessed'
    train_data_percent = 0.85

    # Preprocessing Data
    preprocess(file_path=file_path,
               train_data_percent=train_data_percent,
               saving_path=saving_path)


if __name__ == "__main__":
    _main()
