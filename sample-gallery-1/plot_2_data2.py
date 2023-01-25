"""
Data download example 2
=======================

We can use the same ``iris`` dataset in this example, without downloading it
twice as we know ``data_download`` will check if the data has already been
downloaded.
"""

import pandas as pd
import SampleModule.data_download as dd

data_file = dd.download_data(
    url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',\
    data_file_name='iris.csv',
    data_key='data_dir')

iris = pd.read_csv(data_file, header=None)
iris.head()