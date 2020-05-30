# Naive Basye Based Spam Detector

## Attack

- Changing each letter with its simillar character

## Results

<table style="text-align:center">
    <thead>
        <tr>
            <th>Data</th>
            <th>Num of Samples</th>
            <th>Accuracy (Before)</th>
            <th>Accuracy (After)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Train</td>
            <td>4736</td>
            <td>99.11%</td>
            <td>31.50%</td>
        </tr>
        <tr>
            <td>Test</td>
            <td>836</td>
            <td>99.40%</td>
            <td>30.74%</td>
        </tr>
    </tbody>
</table>

## Data Preprocessing

- Go to dataset directory

```bash
cd dataset_dir
```

- run `preprocess.sh`

```bash
bash preprocess.sh
```

## How to use

- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Anaconda.

```bash
pip install conda
```

- Create enviroment with `spam.yml`.

```bash
conda env create --file envname.yml
```

- Set parameters in `detector.sh`
- Run `detector.sh`

```bash
bash detector.sh
```

- For more help, run:

```bash
python main.py --help
```

## Reference

https://towardsdatascience.com/how-to-build-and-apply-naive-bayes-classification-for-spam-filtering-2b8d3308501
