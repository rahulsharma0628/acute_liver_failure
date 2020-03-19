def tackle_duplicates(self, df):
    """
    :param df: the dataset in the raw form
    :return: gives the unique records in the dataset
    """
    return df[df.duplicated(subset=None, keep='first')]

def count_missing_values(self, df):
    """

    :param df: the dataset
    :return: return the count of null values in the dataset
    """
    return df.isnull().sum()

def check_skewness(self, df):
    """

    :param df: dataframe consisting continuous features
    :return: skewness values of each continuous feature
    """
    return df.skew(skipna=True)

def check_kurtosis(self, df):
    """

    :param df: dataframe consisting continuous features
    :return: kurtosis values of each continuous feature
    """
    for col in df.columns
    print(col + ': ',"{0:.4f}".format(df[col].kurt()))


