def fun_power(x,y):
    """
    Returns x raised to the power of y.

    Parameters
    ----------
    x : int
        base
    y : int
        exponent

    Returns
    -------
    out : int
        the result of x to the power of y
    """

    return x**y

class class_power():
    """A class that performs the power function."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def power(self):
        try:
            return self.x**self.y
        except:
            print('Something went wrong. Make sure x and y are both numbers')
