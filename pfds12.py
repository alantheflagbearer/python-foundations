students = [
    {"name": "Mutasim", "grade": 88, "passed": True},
    {"name": "Ali",     "grade": 45, "passed": False},
    {"name": "Sara",    "grade": 95, "passed": True},
    {"name": "John",    "grade": 38, "passed": False},
    {"name": "Priya",   "grade": 77, "passed": True},
]

#print all student name
for student in students:
    print(student["name"])

#print all students who passed
for student in students:
    if student["passed"]:
        print(student["name"], "passed with grade", student["grade"])

#Calculate the average grade of all students
total_grade = 0
for student in students:
    total_grade = total_grade + student["grade"]
average_grade = total_grade / len(students)
print("Average grade:", average_grade)
