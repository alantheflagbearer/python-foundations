#given this messy sentence
sentence = "   my name is md mutasim billah  "

# Your task — use string methods to:
# 1. Remove the extra spaces
# 2. Make it ALL UPPERCASE
# 3. Replace "mutasim" with your actual name
# 4. Count how many letters 'a' are in the result
# 5. Print each result on its own line

step1 = sentence.strip()
step2 = step1.upper()
step3 = step2.replace("MUTASIM BILLAH", "ALAN")
step4 = step3.count("A")

print(step1)
print(step2)
print(step3)
print(f"letter 'a' appears:{step4} times")