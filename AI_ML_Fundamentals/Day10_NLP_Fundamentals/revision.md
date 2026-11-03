# Day 10 — NLP Quick Revision

## Core definitions

| Term | Remember |
|---|---|
| NLP | AI field for processing human language |
| Corpus | Collection of documents |
| Document | Individual text record |
| Token | Word, subword or character unit |
| Tokenisation | Splitting text into tokens |
| Vocabulary | Collection of unique tokens |
| Preprocessing | Cleaning and normalising text |

## Preprocessing

- Lowercasing: `ACCOUNT` → `account`
- Punctuation removal: `refund!!!` → `refund`
- Stopword removal: task-dependent; preserve negations such as `not`.
- Stemming: approximate root, e.g. `studies` → `studi`.
- Lemmatization: dictionary base form, e.g. `studies` → `study`.

**Tokenisation ≠ Stemming**

Tokenisation splits text; stemming reduces words.

## BoW vs TF-IDF vs Embeddings

| Technique | Main idea | Limitation |
|---|---|---|
| BoW | Word occurrence counts | No context |
| TF-IDF | Weighted word importance | No semantic understanding |
| Embeddings | Dense semantic representations | Quality depends on model and training |

### Formulas

\[
TF=\frac{\text{Term occurrences}}{\text{Total terms in document}}
\]

\[
IDF=\log\left(\frac{N}{DF(t)}\right)
\]

\[
TFIDF=TF\times IDF
\]

Remember:
- TF = frequency within one document.
- IDF = rarity across the corpus.
- Common terms generally receive lower IDF.
- Scikit-learn uses smoothing and normalisation by default.

## Embeddings

- Static: Word2Vec, GloVe, FastText.
- Contextual: Transformer-based representations.
- Token ID = vocabulary identifier.
- Embedding = dense learned vector.
- Similar meanings can have nearby vectors.

## RAG connection

```text
Documents
   ↓
Chunking
   ↓
Embedding generation
   ↓
Vector storage
   ↓
User query embedding
   ↓
Similarity search
   ↓
Relevant context
   ↓
LLM response
```

## Python libraries and methods

| Library / Class | Purpose |
|---|---|
| `re` | Regular expressions |
| `TreebankWordTokenizer` | Tokenisation |
| `PorterStemmer` | Stemming |
| `CountVectorizer` | BoW |
| `TfidfVectorizer` | TF-IDF |
| `fit_transform()` | Learn and transform |
| `get_feature_names_out()` | Retrieve vocabulary |
| `toarray()` | Display sparse matrix as array |

## Important interview distinctions

1. BoW counts words; TF-IDF weights words.
2. Neither BoW nor TF-IDF understands semantic context.
3. Embeddings capture learned relationships between words.
4. Contextual embeddings depend on surrounding text.
5. Subword tokenisation handles rare and unfamiliar words.
6. Stopword removal can change sentence meaning.
7. Static embeddings and contextual embeddings are different.

## Self-assessment

- [x] Explain NLP, corpus, document and token.
- [x] Explain preprocessing and tokenisation.
- [x] Differentiate stemming and lemmatisation.
- [x] Explain BoW and its limitations.
- [x] Explain TF, IDF and TF-IDF.
- [x] Explain static and contextual embeddings.
- [x] Differentiate token IDs and embeddings.
- [x] Explain how embeddings support RAG.
- [x] Implement preprocessing using NLTK.
- [x] Implement BoW and TF-IDF using scikit-learn.

## Day 10 result

**Final assessment: 6/6**

Practical files:
- `practical/text_preprocessing.py`
- `practical/text_vectorization.py`

Status: Completed.