#Count how many times each word appears in a sentence. This exact pattern is used in real NLP and data science work.
sentence = "the cat sat on the mat the cat"
#Split the sentence into words
words = sentence.split()

#Create an empty dictionary to store the word counts
word_counts = {}

#Loop through each word in the list of words
for word in words:
    if word in word_counts:
        word_counts[word] = word_counts[word] + 1
    else:
        word_counts[word] = 1
#Print the word counts
print(word_counts)
