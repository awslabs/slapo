"""
SampleModule example
====================

This example will demonstrate the ``power`` function and ``class_power`` from
our package 'SampleModule'.
"""

import SampleModule.module

SampleModule.module.fun_power(2,3)

#%%
# The function ``power`` returns the first number raised to the power of the
# second number.

my_class = SampleModule.module.class_power(2,3)
my_class.power()

#%%
# Instances of ``power_class`` have a ``power`` method. If a non-number is
# used when initiating the class, an informative statement is printed:

my_class = SampleModule.module.class_power('a',3)
my_class.power()
