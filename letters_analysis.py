import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _main():
    letters_pkl_file = './datasets/SMS_Spam_Collection_Dataset/preprocessed/letters.pkl'
    letters = pd.read_pickle('{}'.format(letters_pkl_file))

    xs = list()
    ys = list()
    total_num = 0
    for ix, row in letters.iterrows():
        xs.append(row['letter'])
        ys.append(row['num'])
        total_num += row['num']

    ys = (np.array(ys) / total_num)*100
    plt.plot([ix for ix in range(len(xs))], ys)
    plt.xticks([ix for ix in range(len(xs))], xs)
    plt.grid()
    plt.xlabel('Letters')
    plt.ylabel('Percent(%)')
    plt.title('Letters Distribution')
    plt.show()


if __name__ == "__main__":
    _main()
