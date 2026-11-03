from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "fraud detected in bank transaction",
    "customer requested account refund",
    "fraud reported in account transaction",
    "customer reported a failed transaction"
]

# 1. Create a CountVectorizer
count_vectorizer = CountVectorizer()

# 2. Fit and transform the documents
count_matrix = count_vectorizer.fit_transform(documents)

# 3. Print the vocabulary
print("Vocabulary:", count_vectorizer.get_feature_names_out())

# 4. Print the Bag of Words matrix
print("Bag of Words Matrix:")
print(count_matrix.toarray())

# 5. Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# 6. Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 7. Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())