import pandas as pd
from collections import Counter
import string

test = pd.read_csv("./Data/spam/spam_train1.csv")

ham_count = 0
spam_count = 0

not_spam_word_count = 0
spam_word_count = 0

ham_word_counter = Counter()
spam_word_counter = Counter()

def clean(text):
    return str(text).translate(str.maketrans("", "", string.punctuation)).lower().split


for i in range(len(test)):
    if (test.iloc[i, 0] == 'ham'):
        ham_count += 1
        words = clean(test.iloc[i,1])
        not_spam_word_count += len(words)
        ham_word_counter.update(words)
    else:
        spam_count += 1
        words = clean(test.iloc[i,1])
        spam_word_count += len(words)
        spam_word_counter.update(words)

print(f"Not Spam: {ham_count}")
print(f"Spam: {spam_count}")

print(f"Total Word Count for Not Spam: {not_spam_word_count}")
print(f"Total Word Count for Spam: {spam_word_count}")

for word in ham_word_counter:
    ham_word_counter[word] /= not_spam_word_count
    ham_word_counter[word] = "{:.4f}".format(ham_word_counter[word])

for word in spam_word_counter:
    spam_word_counter[word] /= spam_word_count
    spam_word_counter[word] = "{:.4f}".format(spam_word_counter[word])


df = pd.DataFrame(list(ham_word_counter.items()), columns=["word", "count"])
df.to_csv("./Calculations/ham_count.csv", index=False)

df = pd.DataFrame(list(spam_word_counter.items()), columns=["word", "count"])
df.to_csv("./Calculations/spam_count.csv", index=False)



    