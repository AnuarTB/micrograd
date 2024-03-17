from micrograd import Value

a = Value(3)
b = Value(4)
c = a * b
print(c.data)

d = Value(4)
e = 2 + d
print(e.data)

f = Value(5)
g = 3 - f
print(g.data)

h = Value(4)
i = 9 / h
print(i.data)
