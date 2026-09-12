#Loop through a list of scores. Print each one, calculate the total, and find the highest score without using max()

scores = [72, 88, 65, 95, 81, 77]

total = 0
highest = 0

for score in scores:
    print(f"Score: {score}")
    total = total + score
    if score > highest:
        highest = score

average = total / len(scores)
print(f"Average: {average}")
print(f"Highest: {highest}")
print(f"Total: {total}")
