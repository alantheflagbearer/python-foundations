def calculate_average(my_scores):
    total = sum(my_scores)
    count = len(my_scores)
    average = total / count
    return average

#test it - always test right after writng
my_scores = [72, 88, 65, 95, 81, 77]
result = calculate_average(my_scores)
print(f"average: {result}")
print(f"average: {result: .2f}")