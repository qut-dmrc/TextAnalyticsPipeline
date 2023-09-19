'''
Uses spaCy, NLTK or Stanza process to:
    - detect entities in a body of text (Named Entity Recognition)
    - detect parts of speech in a body of text (Part of Speech Tagging)
    - parse the linguistic structure of a body of text (Dependency Parsing)

Documentation:
'''

from TextAnalyticsPipeline.perform_analysis import run_text_pipeline

if __name__ == "__main__":
    run_text_pipeline()
