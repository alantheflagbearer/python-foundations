
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

