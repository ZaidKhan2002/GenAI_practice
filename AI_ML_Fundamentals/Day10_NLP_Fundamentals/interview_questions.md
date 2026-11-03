# Day 10 — NLP Interview Questions

## NLP Fundamentals

### 1. What is NLP?

NLP is a field of AI that enables computers to process, analyse, understand and generate human language.

### 2. What is the difference between a corpus and a document?

A corpus is a collection of documents, whereas a document is one individual text record.

### 3. What is tokenisation?

Tokenisation is the process of breaking text into smaller units called tokens. These may be words, subwords or characters.

### 4. Why is text preprocessing required?

Raw text may contain inconsistent casing, punctuation, unnecessary whitespace and other noise. Preprocessing helps standardise the input for downstream NLP tasks.

### 5. Why do LLMs use subword tokenisation?

Subword tokenisation helps handle rare and unfamiliar words, prefixes and suffixes while keeping the vocabulary manageable.

## Preprocessing

### 6. What is the difference between stemming and lemmatization?

Stemming uses approximate rules to reduce words to stems, which may not be dictionary words. Lemmatization uses linguistic knowledge and grammatical context to produce valid base forms.

### 7. Why should stopwords not always be removed?

Some stopwords carry important meaning. For example, removing `not` from `I did not receive my refund` reverses the meaning.

### 8. What is the difference between tokenisation and stemming?

Tokenisation splits text into tokens. Stemming reduces individual words to approximate root forms.

## Bag of Words and TF-IDF

### 9. What is Bag of Words?

BoW represents a document using a numerical vector containing the occurrence count of each vocabulary word.

### 10. What are the limitations of BoW?

- Ignores word order.
- Does not understand semantic relationships.
- Can generate high-dimensional sparse vectors.

### 11. What is TF-IDF?

TF-IDF is a statistical technique that assigns importance weights to words based on their frequency in a document and their rarity across the corpus.

### 12. Explain TF and IDF.

- TF measures how frequently a term occurs in a document.
- IDF measures how rare the term is across the collection of documents.

### 13. Why does TF-IDF downweight common words?

Words appearing in many documents provide less distinguishing information. IDF reduces their influence.

### 14. What is the difference between BoW and TF-IDF?

BoW uses raw word counts, whereas TF-IDF uses weighted word values based on document frequency.

### 15. Does TF-IDF understand semantic similarity?

No. TF-IDF is a statistical weighting technique. It does not inherently understand meaning or context.

## Word Embeddings

### 16. What are word embeddings?

Word embeddings are dense numerical vectors that represent words and capture learned semantic and linguistic relationships.

### 17. What is the difference between static and contextual embeddings?

Static embeddings generally assign the same representation to a word regardless of context. Contextual embeddings generate representations influenced by surrounding words.

### 18. Give examples of static embedding techniques.

Word2Vec, GloVe and FastText.

### 19. What is the difference between a token ID and an embedding?

A token ID is an identifier in the tokenizer's vocabulary. An embedding is a learned dense vector representing the token.

### 20. Why are embeddings useful in semantic search?

They allow text to be represented in a vector space where semantically similar content can be retrieved using similarity measures such as cosine similarity.

## LLMs and RAG

### 21. How does traditional NLP differ from LLM-based NLP?

Traditional NLP often uses manually selected or statistically derived features such as BoW and TF-IDF with task-specific models. LLM-based NLP uses pretrained Transformer architectures and contextual representations to perform a broader range of language tasks.

### 22. How are embeddings used in RAG?

Documents and queries are converted into embeddings. A retrieval system compares their vector representations to find relevant document chunks, which are then passed to an LLM as context.

### 23. What is cosine similarity?

Cosine similarity measures the cosine of the angle between two vectors. It is commonly used to estimate the directional similarity of embeddings.

### 24. Are embeddings always better than TF-IDF?

No. TF-IDF may be sufficient for keyword-oriented tasks and can be computationally cheaper. Embeddings are more useful when semantic relationships and different wording matter.

## Practical and implementation questions

### 25. What is CountVectorizer?

CountVectorizer is a scikit-learn class that builds a vocabulary and converts text documents into word-count matrices.

### 26. What is TfidfVectorizer?

TfidfVectorizer converts text documents into TF-IDF weighted numerical matrices.

### 27. What does fit_transform() do?

It learns the vocabulary or statistical parameters from the supplied data and transforms the documents into numerical features.

### 28. Why are BoW and TF-IDF matrices sparse?

Most documents contain only a small subset of the total vocabulary, so most matrix entries are zero.

### 29. Why does scikit-learn's TF-IDF output differ from a simple manual calculation?

TfidfVectorizer applies IDF smoothing and L2 normalisation by default, among other configurable options.

### 30. What is the role of embeddings in a Transformer?

An embedding layer converts token IDs into dense vectors that the Transformer can process. Contextual representations are then built through the Transformer layers.