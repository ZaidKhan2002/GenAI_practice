import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

text = """
My ACCOUNT has been DEBITED twice!!!
The customer was complaining about duplicate transactions.
"""

# Step 1: Lowercase
cleaned_text = text.lower()

# Step 2: Remove punctuation
cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)

# Step 3: Tokenisation
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(cleaned_text)

# Step 4: Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens]

print("Original text:", text)
print("\nCleaned text:", cleaned_text)
print("\nTokens:", tokens)
print("\nStemmed tokens:", stemmed_tokens)
