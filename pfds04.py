# Part 1 — guess the type before running
a = "26"
b = 26
c = 26.0
d = True

# Run this to check — what do you expect?
print(type(a))
print(type(b))
print(type(c))
print(type(d))

itgr = int(a)
result = itgr + b
print(f"String + int = {result}")