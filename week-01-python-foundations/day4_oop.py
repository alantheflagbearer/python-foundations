class Dog:

    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def bark(self):
        return f"{self.name} says: Woof!"


rex = Dog("Rex", "Labrador")
print(rex.name)
print(rex.bark())



#method function day4 student class report

class Student:
    """Represents a student with a name and list of scores."""

    def __init__(self, name, scores):
        """Initialize student with name and scores list."""
        self.name = name
        self.scores = scores

    def average(self):
        """Return the average score."""
        return sum(self.scores) / len(self.scores)

    def passed(self, passing_score=50):
        """Return True if average >= passing_score."""
        return self.average() >= passing_score

    def highest_score(self):
        """Return the highest individual score."""
        return max(self.scores)

    def report(self):
        """Print a formatted report card."""
        print("=" * 35)
        print(f"  REPORT CARD — {self.name.upper()}")
        print("=" * 35)
        for i, score in enumerate(self.scores, 1):
            print(f"  Subject {i}: {score}/100")
        print("-" * 35)
        print(f"  Average : {self.average():.1f}/100")
        print(f"  Result  : {'PASS' if self.passed() else 'FAIL'}")
        print("=" * 35)


# ── TESTS ────────────────────────────────────
mutasim = Student("Mutasim", [80, 58, 77])
sara    = Student("Sara",    [95, 90, 88])
ali     = Student("Ali",     [40, 45, 38])
mutasim.report()
print()
sara.report()
print()
ali.report()
print()
print(f"Mutasim average: {mutasim.average():.1f}")
print(f"Sara passed:     {sara.passed()}")
print(f"Ali passed:      {ali.passed()}")
print(f"Ali passed(60):  {ali.passed(passing_score=60)}")



#class classroom day4
class Classroom:
    """Represents a classroom holding multiple Student objects."""

    def __init__(self):
        """Initialize with an empty list of students."""
        self.students = []

    def add_student(self, student):
        """Add a Student object to the classroom."""
        self.students.append(student)

    def class_average(self):
        """Return the average score across all students."""
        if not self.students:
            return 0
        total = sum(s.average() for s in self.students)
        return total / len(self.students)

    def top_student(self):
        """Return the student with the highest average."""
        return max(self.students, key=lambda s: s.average())

    def show_all_reports(self):
        """Print report cards for all students."""
        for student in self.students:
            student.report()
            print()

#---test--------------
classroom = Classroom()
classroom.add_student(mutasim)
classroom.add_student(sara)
classroom.add_student(ali)

print("*" * 35)
print(f"class average: {classroom.class_average():.1f}")
best = classroom.top_student()
print(f"top student: {best.name} ({best.average():.1f})")

#Refactor pfds12 using your new classes
#Turn the old dict-based student list into proper objects

# ── REFACTOR pfds12 ──────────────────────────
print("\n=== FULL CLASS REPORT ===")

pfds12_class = Classroom()
pfds12_class.add_student(Student("Mutasim", [88, 80, 77]))
pfds12_class.add_student(Student("Ali",     [45, 50, 40]))
pfds12_class.add_student(Student("Sara",    [95, 92, 98]))
pfds12_class.add_student(Student("John",    [38, 42, 35]))
pfds12_class.add_student(Student("Priya",   [77, 80, 75]))

pfds12_class.show_all_reports()

print("+" * 35)
print(f"class average ; {pfds12_class.class_average():.1f}")
best = pfds12_class.top_student()
print(f"Top student   : {best.name} ({best.average():.1f})")

#print only students who passed
print("\nstudents who passed:")
for s in pfds12_class.students:
    if s.passed():
        print(f" {s.name} - {s.average():.1f}")