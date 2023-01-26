"""
Data download example
=====================

This example shows one way of dealing with large data files required for your
examples.

The ``download_data`` function first checks if the data has already been
downloaded, looking in either the data directory saved the configuration file
(by default ``~/.sg_template``) or the default data directory. If the data has
not already been downloaded, it downloads the data from the url and saves the
data directory to the configuration file. This allows you to use the data
again in a different example without downloading it again.

Note that examples in the gallery are ordered according to their filenames, thus
the number after 'plot\_' dictates the order the example appears in the gallery.
"""

import pandas as pd
import SampleModule.data_download as dd


data_file = dd.download_data(
    url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',\
    data_file_name='iris.csv',
    data_key='data_dir')

iris = pd.read_csv(data_file, header=None)
iris.head()






