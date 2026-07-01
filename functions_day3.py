
def grade_report(name, scores, passing_score=50):
    """Print a formatted report card for a student.

    Args:
        name (str): student name
        scores (list): list of numeric scores
        passing_score (int): minimum average to pass. Default 50.

    Returns:
        None — prints the report card
    """
    average = sum(scores) / len(scores)  # fixed bug from Day 1!
    passed = average >= passing_score

    print("=" * 35)
    print(f"  REPORT CARD — {name.upper()}")
    print("=" * 35)
    for i, score in enumerate(scores, 1):
        print(f"  Subject {i}: {score}/100")
    print("-" * 35)
    print(f"  Average : {average:.1f}/100")
    print(f"  Result  : {'PASS' if passed else 'FAIL'}")
    print("=" * 35)


# ── TESTS ────────────────────────────────────
grade_report("Mutasim", [80, 58, 77])           # default passing=50
print()
grade_report("Ali", [90, 95, 88])
print()
grade_report("Sara", [40, 45, 38], passing_score=60)  # stricter school

#count_words function day 3
def count_words(sentence):
    """Count how many times each word appears in a sentence.

    Args:
        sentence (str): any string of text

    Returns:
        dict: word → count pairs, e.g. {'the': 3, 'cat': 2}
    """
    words = sentence.lower().split()  # lowercase first so "The" == "the"
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts


# ── TESTS ────────────────────────────────────
s1 = "the cat sat on the mat the cat"
s2 = "data science is fun data is great"
s3 = "I love Python and Python loves data"

print(count_words(s1))  # {'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1}
print(count_words(s2))
print(count_words(s3))

#Write top_student() day3
def top_student(students):
    """Find the student with the highest grade.

    Args:
        students (list): list of dicts, each with 'name' and 'grade' keys

    Returns:
        dict: the student dict with the highest grade
    """
    best = students[0]
    for student in students:
        if student["grade"] > best["grade"]:
            best = student
    return best

#── TEST with pfds12 data ────────────────────
students = [  {"name": "Mutasim", "grade": 88, "passed": True},
    {"name": "Ali",     "grade": 45, "passed": False},
    {"name": "Sara",    "grade": 95, "passed": True},
    {"name": "John",    "grade": 38, "passed": False},
    {"name": "Priya",   "grade": 77, "passed": True},]

winner = top_student(students)
print(f" top student: {winner['name']} with {winner['grade']}/100")


#wordfrequency function day3
def word_frequency_sorted(sentence):
    """Count words and return sorted from most to least frequent.

    Args:
        sentence (str): any string of text

    Returns:
        list of tuples: [(word, count), ...] sorted by count descending
    """

    counts = count_words(sentence)
    sorted_counts = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True
     )

    return sorted_counts
#test-------------
sentence = "the cat sat on the mat the cat sat"
result = word_frequency_sorted(sentence)
print("\nword frequencies (most to least):")
for word, count in result:
    print(f" {word:10} : {count}")