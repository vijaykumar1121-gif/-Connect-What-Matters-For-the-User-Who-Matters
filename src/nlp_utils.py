import nltk
from typing import List
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob

# Download punkt if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('stopwords') # Added for keyword extraction

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load a small, efficient sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load HuggingFace pipelines (optimized for 1GB constraint)
summarizer = pipeline('summarization', model='t5-small', tokenizer='t5-small', framework='pt')
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def sentence_segment(text: str) -> List[str]:
    """Split text into sentences using nltk."""
    return nltk.sent_tokenize(text)

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Simple keyword extraction using word frequency (can be replaced with RAKE/KeyBERT)."""
    from collections import Counter
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    most_common = Counter(keywords).most_common(top_n)
    return [w for w, _ in most_common]

def get_embedding(text: str):
    """Get a sentence embedding for the given text."""
    return model.encode(text)

def semantic_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using embeddings."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return float(util.pytorch_cos_sim(emb1, emb2).item())

def extract_entities(text: str, entity_types: list = None) -> list:
    """Extract named entities from text using spaCy. Optionally filter by entity types."""
    doc = nlp(text)
    if entity_types:
        return [ent.text for ent in doc.ents if ent.label_ in entity_types]
    else:
        return [ent.text for ent in doc.ents]

# --- Topic Modeling (Alternative to gensim) ---
def extract_topics(texts: list, num_topics: int = 3, num_words: int = 5):
    """Extract topics from a list of texts using scikit-learn LDA."""
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(tfidf_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append((topic_idx, ' + '.join(top_words)))
        
        return topics
    except Exception as e:
        # Fallback: return simple word frequency
        print(f"Topic modeling failed: {e}")
        all_text = ' '.join(texts)
        keywords = extract_keywords(all_text, num_words * num_topics)
        return [(i, ' + '.join(keywords[i*num_words:(i+1)*num_words])) 
                for i in range(num_topics)]

# --- Abstractive Summarization ---
def abstractive_summary(text: str, max_length: int = 60, min_length: int = 20) -> str:
    """Generate an abstractive summary using a transformer model."""
    if len(text.split()) < min_length:
        return text
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Fallback: return first few sentences
        sentences = sentence_segment(text)
        return ' '.join(sentences[:3])

# --- Intent Classification (Rule-based) ---
def classify_intent(job_text: str) -> str:
    """Classify the intent of the job-to-be-done (summarize, analyze, compare, extract, etc.)."""
    job_text = job_text.lower()
    if any(word in job_text for word in ['summarize', 'summary', 'overview', 'review']):
        return 'summarize'
    elif any(word in job_text for word in ['analyze', 'analysis', 'examine', 'investigate']):
        return 'analyze'
    elif any(word in job_text for word in ['compare', 'contrast', 'difference']):
        return 'compare'
    elif any(word in job_text for word in ['extract', 'find', 'identify', 'list']):
        return 'extract'
    else:
        return 'general'

# --- Cross-Document Entity Aggregation ---
def aggregate_entities(docs: list) -> dict:
    """Aggregate named entities across a list of texts."""
    from collections import defaultdict
    entity_map = defaultdict(list)
    for idx, text in enumerate(docs):
        for ent in extract_entities(text):
            entity_map[ent].append(idx)
    return dict(entity_map)

# --- Cross-Document Topic Aggregation ---
def aggregate_topics(docs: list, num_topics: int = 3, num_words: int = 5):
    """Aggregate topics across a list of texts."""
    return extract_topics(docs, num_topics=num_topics, num_words=num_words)

# --- AI-Powered Question Answering ---
def answer_question(question: str, context: str) -> str:
    """Answer a question given a context using a QA model."""
    try:
        result = qa_pipeline({'question': question, 'context': context})
        return result['answer']
    except Exception as e:
        return f"Unable to answer question: {e}"

# --- Zero-Shot Classification (Rule-based to stay within 1GB constraint) ---
def zero_shot_classify(text: str, labels: list) -> dict:
    """Classify text into arbitrary labels using lightweight rule-based approach."""
    text_lower = text.lower()
    scores = {}
    for label in labels:
        score = 0.0
        if label == "methods":
            keywords = ["method", "approach", "algorithm", "procedure", "technique", "protocol"]
        elif label == "results":
            keywords = ["result", "finding", "outcome", "performance", "accuracy", "score"]
        elif label == "limitations":
            keywords = ["limitation", "constraint", "drawback", "weakness", "issue", "problem"]
        elif label == "conclusion":
            keywords = ["conclusion", "summary", "final", "overall", "therefore", "thus"]
        elif label == "introduction":
            keywords = ["introduction", "background", "overview", "context", "motivation"]
        else:
            keywords = [label]

        for keyword in keywords:
            if keyword in text_lower:
                score += 1.0
        scores[label] = min(score / len(keywords), 1.0)
    return scores

# --- Sentiment and Subjectivity Analysis ---
def sentiment_subjectivity(text: str) -> dict:
    """Analyze sentiment and subjectivity using TextBlob."""
    blob = TextBlob(text)
    return {'sentiment': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity} 