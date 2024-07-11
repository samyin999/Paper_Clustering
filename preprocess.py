import os
import re
from pdfminer.high_level import extract_text
import spacy

def extract_abstract(path):
    try:
        text = extract_text(path).lower().split('\n')
        abstract_content = ""
        tokens = []

        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        for line in text:
            if len(line) == 0:
                cleaned_content = re.sub(r'[^a-z\s]', '', abstract_content)
                tokens = nlp(cleaned_content)
                
                if len(tokens) > 60 and len(tokens) < 350:
                    break
                else:
                    abstract_content = ""
            else:
                abstract_content += line + " "

        lemmas = [token.lemma_ for token in tokens if not token.is_stop]
        return lemmas
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return []

def process_papers(directory):
    papers_tokens = []
    paper_paths = []

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            path = os.path.join(directory, filename)
            paper_paths.append(path)

    for path in paper_paths:
        print(f"Processing: {path}")
        tokens = extract_abstract(path)
        if tokens:
            papers_tokens.append(tokens)
            print(f"Extracted {len(tokens)} tokens")
        else:
            print(f"No tokens extracted from {path}")

    return papers_tokens

if __name__ == "__main__":
    paper_directory = './papers/'
    processed_papers = process_papers(paper_directory)
    print(f"Processed {len(processed_papers)} papers")