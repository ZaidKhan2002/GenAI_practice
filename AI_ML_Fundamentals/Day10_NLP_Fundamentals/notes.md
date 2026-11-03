# Day 10 — NLP Fundamentals

## 1. What is NLP?

Natural Language Processing (NLP) is a field of AI that enables computers to process, analyse, understand and generate human language.

**Applications:**
- Chatbots and virtual assistants
- Spam detection
- Sentiment analysis
- Language translation
- Document classification
- Semantic search

### Important terminology

- **Corpus:** Collection of text documents.
- **Document:** One individual text record.
- **Token:** A unit of text, such as a word, subword or character.
- **Tokenisation:** Breaking text into tokens.
- **Vocabulary:** Collection of unique tokens recognised by a vectorizer or tokenizer.

Traditional ML models generally require numerical features, so text is converted into numerical representations before being processed.

## 2. Text preprocessing

Text preprocessing cleans and normalises raw text before further processing.

Common operations:

1. Lowercasing: `ACCOUNT` → `account`
2. Punctuation removal: `refund!!!` → `refund`
3. Whitespace normalisation.
4. Stopword removal, when appropriate.
5. Stemming.
6. Lemmatization.

**Important:** Stopwords should not always be removed. Removing `not` from `I did not receive my refund` changes the meaning.

### Tokenisation

Breaking text into smaller units called tokens.

Example:

```python
text = "My account has been debited"

tokens = ["My", "account", "has", "been", "debited"]
```

LLMs often use subword tokenisation to handle rare and unfamiliar words while maintaining a manageable vocabulary.

### Stemming vs Lemmatization

| Stemming | Lemmatization |
|---|---|
| Uses approximate rules | Uses linguistic knowledge |
| Removes word suffixes | Finds dictionary base forms |
| May produce invalid words | Produces valid lemmas |
| Usually faster | More linguistically informed |

Examples:

- `studies` → `studi` (stemming)
- `studies` → `study` (lemmatisation)
- `was` → `be` (lemmatisation, depending on context)

Python libraries:
- NLTK PorterStemmer
- NLTK WordNetLemmatizer

## 3. Bag of Words (BoW)

Bag of Words converts text into numerical vectors based on word occurrence counts.

Example:

```text
Document 1: I love AI
Document 2: I love NLP
```

Vocabulary:

```python
["I", "love", "AI", "NLP"]
```

Vectors:

```python
Document 1: [1, 1, 1, 0]
Document 2: [1, 1, 0, 1]
```

The values represent how often each vocabulary word occurs in a document.

**Limitations:**
- Ignores word order.
- Does not capture semantic meaning.
- Can generate high-dimensional sparse vectors.

Python implementation:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(bow_matrix.toarray())
```

## 4. TF-IDF

Term Frequency–Inverse Document Frequency assigns weights to words based on their frequency in a document and their rarity across the corpus.

### Term Frequency (TF)

Measures how frequently a word occurs in one document.

\[
TF(t,d)=\frac{\text{Occurrences of }t}{\text{Total words in }d}
\]

### Inverse Document Frequency (IDF)

Measures how rare a word is across the corpus.

\[
IDF(t)=\log\left(\frac{N}{DF(t)}\right)
\]

Where:
- N = total number of documents.
- DF(t) = number of documents containing the term.

### TF-IDF

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

A word that appears frequently in one document but rarely across other documents generally receives a higher weight.

**Example:** A term such as `fraud` may be more informative than a common term such as `the`.

Python implementation:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
```

Note: Scikit-learn applies IDF smoothing and L2 normalisation by default, so its values may differ from simplified manual calculations.

## 5. Word embeddings

Word embeddings represent words as dense numerical vectors that capture learned semantic and linguistic relationships.

Example:

```python
car = [0.21, 0.87, -0.34, 0.62, ...]
```

Words with similar meanings tend to have nearby representations in embedding space.

### Types of embeddings

**Static embeddings:**
- Word2Vec
- GloVe
- FastText

A word generally has the same representation regardless of its context.

**Contextual embeddings:**

Transformer-based models generate representations that depend on surrounding context.

Example:

- Money deposited in a bank.
- Sitting near a river bank.

The representation of `bank` can differ depending on context.

### Token IDs vs embeddings

- Token ID: Numerical identifier assigned to a token.
- Embedding: Dense vector representing that token.

Simplified flow:

```text
Input text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embedding layer
   ↓
Dense vectors
   ↓
Transformer
```

## 6. Traditional NLP vs LLM-based NLP

| Traditional NLP | LLM-based NLP |
|---|---|
| BoW, TF-IDF, static embeddings | Subword tokens and learned embeddings |
| Often task-specific | Broad pretrained capabilities |
| Usually lower compute requirements | Usually higher compute requirements |
| Often limited context modelling | Contextual representations |
| Examples: spam classifier | Examples: conversational assistant |

Traditional NLP can be preferable for simple, well-defined tasks because it is efficient and easier to interpret. LLMs are more flexible but require evaluation, security controls and cost management.

## 7. Why embeddings matter in RAG

Embeddings are a foundation of semantic retrieval.

Simplified RAG flow:

1. Split documents into chunks.
2. Generate embeddings for each chunk.
3. Store embeddings in a vector database or index.
4. Generate an embedding for the user's query.
5. Retrieve semantically similar chunks.
6. Pass retrieved context to an LLM.
7. Generate an answer grounded in the retrieved information.

Example:

Query: `How can I reverse a failed transaction?`

Knowledge-base content: `Procedure for handling unsuccessful payment debits.`

Even though the wording differs, embeddings can help identify the semantic relationship.

## 8. Practical implementation completed

Repository files:

```text
10-NLP-Fundamentals/
└── practical/
    ├── text_preprocessing.py
    └── text_vectorization.py
```

### text_preprocessing.py

Implemented:
- Lowercasing
- Punctuation removal using regex
- Tokenisation using NLTK TreebankWordTokenizer
- Stemming using PorterStemmer

### text_vectorization.py

Implemented:
- CountVectorizer
- Vocabulary extraction
- Bag of Words matrix
- TfidfVectorizer
- TF-IDF matrix

### Key observation

BoW represents word counts, while TF-IDF represents weighted word importance. Both are sparse numerical representations and do not inherently understand semantic context.

## Day 10 conclusion

NLP transforms human language into representations that machines can process. Traditional approaches such as BoW and TF-IDF use statistical features, while embeddings provide dense representations that capture learned relationships. Contextual embeddings are central to modern Transformer-based language models and semantic retrieval.