def myFunc():
    yield "Hello"
    yield 51
    yield "Good Bye"

x = myFunc()
print("type of x = ", type(x))
# Use x in a loop
for z in x:
    print(z)
