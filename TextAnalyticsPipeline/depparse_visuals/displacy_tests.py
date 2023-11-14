'''
Produces a visualisation of the spaCy dependency parse tree of a sentence.
'''

import spacy
from spacy import displacy

# One document per page
nlp = spacy.load("en_core_web_sm")
doc = nlp("In case you have n't already heard the news, Pauline is coming to WA!")
displacy.serve(doc, style="dep", auto_select_port=True, options={'compact': True})



# Multiple documents in one page
nlp = spacy.load("en_core_web_sm")
doc1 = nlp("This is a sentence.")
doc2 = nlp("This is another sentence.")
displacy.serve([doc1, doc2], style="dep", page=True)

