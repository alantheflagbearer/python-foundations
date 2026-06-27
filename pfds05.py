#mini student card
student_name = "mutasim billah"
subject1_name = "Python"
subject1_score = 80      # give yourself a score out of 100
subject2_name = "Math"
subject2_score = 58
subject3_name = "Data Science"
subject3_score = 77

# Calculate average — add all 3 and divide by 3
average = (subject1_score + subject2_score + subject3_score) / 3

# Boolean: did the student pass? (pass = average >= 50)
passed = average >= 50
# Print the report card
print("=" * 35)
print(f"  REPORT CARD — {student_name.upper()}")
print("=" * 35)
print(f"  {subject1_name}: {subject1_score}/100")
print(f"  {subject2_name}: {subject2_score}/100")
print(f"  {subject3_name}: {subject3_score}/100")
print("-" * 35)
print(f"  Average: {average:.1f}/100")
print(f"  Result: {'PASS' if passed else 'FAIL'}")
print("=" * 35)