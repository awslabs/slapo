"""
First example of gallery 2
==========================

Here we will provide some different examples that show how Sphinx-Gallery can
capture figures output by Matplotlib and html representations of objects,
if present.
"""

import matplotlib.pyplot as plt

_ = plt.plot([1,2,3])

#%%
# pandas dataframes have a html representation, and this is captured:

import pandas as pd

df = pd.DataFrame({'col1': [1,2,3],
                   'col2': [4,5,6]})
df

s = pd.Series([1,2,3])
