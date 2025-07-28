import nltk
from typing import List
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline('summarization', model='t5-small', tokenizer='t5-small', framework='pt')
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
def sentence_segment(text: str) -> List[str]:
    return nltk.sent_tokenize(text)
def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    from collections import Counter
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    most_common = Counter(keywords).most_common(top_n)
    return [w for w, _ in most_common]
def get_embedding(text: str):
    return model.encode(text)
def semantic_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return float(util.pytorch_cos_sim(emb1, emb2).item())
def extract_entities(text: str, entity_types: list = None) -> list:
    doc = nlp(text)
    if entity_types:
        return [ent.text for ent in doc.ents if ent.label_ in entity_types]
    else:
        return [ent.text for ent in doc.ents]
def extract_topics(texts: list, num_topics: int = 3, num_words: int = 5):
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append((topic_idx, ' + '.join(top_words)))
        return topics
    except Exception as e:
        print(f"Topic modeling failed: {e}")
        all_text = ' '.join(texts)
        keywords = extract_keywords(all_text, num_words * num_topics)
        return [(i, ' + '.join(keywords[i*num_words:(i+1)*num_words])) 
                for i in range(num_topics)]
def abstractive_summary(text: str, max_length: int = 60, min_length: int = 20) -> str:
    if len(text.split()) < min_length:
        return text
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        sentences = sentence_segment(text)
        return ' '.join(sentences[:3])
def classify_intent(job_text: str) -> str:
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
def aggregate_entities(docs: list) -> dict:
    from collections import defaultdict
    entity_map = defaultdict(list)
    for idx, text in enumerate(docs):
        for ent in extract_entities(text):
            entity_map[ent].append(idx)
    return dict(entity_map)
def aggregate_topics(docs: list, num_topics: int = 3, num_words: int = 5):
    return extract_topics(docs, num_topics=num_topics, num_words=num_words)
def answer_question(question: str, context: str) -> str:
    try:
        result = qa_pipeline({'question': question, 'context': context})
        return result['answer']
    except Exception as e:
        return f"Unable to answer question: {e}"
def zero_shot_classify(text: str, labels: list) -> dict:
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
def sentiment_subjectivity(text: str) -> dict:
    blob = TextBlob(text)
    return {'sentiment': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity} 