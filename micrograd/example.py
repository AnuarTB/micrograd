from micrograd import Value

a = Value(1)
b = Value(2)

c = a + b
print(c.data)