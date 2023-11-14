'''
Produces a visualisation of the spaCy dependency parse tree of a sentence.
'''

import spacy
from spacy import displacy

# One document per page
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")
displacy.serve(doc, style="dep", options={'compact': True})
print('http://localhost:5000')


# Multiple documents in one page
nlp = spacy.load("en_core_web_sm")
doc1 = nlp("This is a sentence.")
doc2 = nlp("This is another sentence.")
displacy.serve([doc1, doc2], style="dep", page=True)

