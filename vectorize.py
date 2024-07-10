from sklearn.feature_extraction.text import TfidfVectorizer
# Comment out the gensim import
# from gensim.models import Word2Vec
import numpy as np

def tfidf_vectorize(papers_tokens):
    paper_texts = [' '.join(tokens) for tokens in papers_tokens]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(paper_texts)
    return tfidf_matrix, vectorizer

# Doesn't work 
'''
def word2vec_vectorize(papers_tokens, vector_size=100):
    model = Word2Vec(papers_tokens, vector_size=vector_size, window=5, min_count=1, workers=4)
    paper_vectors = [np.mean([model.wv[word] for word in paper if word in model.wv], axis=0) for paper in papers_tokens]
    return np.array(paper_vectors), model
'''

def vectorize_papers(papers_tokens, method='tfidf'):
    if method == 'tfidf':
        return tfidf_vectorize(papers_tokens)
    elif method == 'word2vec':
        raise NotImplementedError("Word2Vec is currently disabled")
    else:
        raise ValueError("Unsupported vectorization method")

if __name__ == "__main__":
    # This is just for testing
    from preprocess import process_papers
    paper_directory = './papers/'
    processed_papers = process_papers(paper_directory)
    
    tfidf_matrix, _ = vectorize_papers(processed_papers, method='tfidf')
    
    print("TF-IDF matrix shape:", tfidf_matrix.shape)