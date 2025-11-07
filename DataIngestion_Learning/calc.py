def add(x,y):
    return x + y 

def multiply(x,y):
    return x * y 

def divide(x,y):
    if y == 0:
        raise ValueError("can't be divided by zero")
    return x/y