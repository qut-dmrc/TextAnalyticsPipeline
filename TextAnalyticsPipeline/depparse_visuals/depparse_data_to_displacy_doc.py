'''
This script takes dependency data from a dataframe and converts it into a spaCy displaCy doc object in order to visualise the dependency parse tree.
Note that this approach assumes you have the word text, head text, and dependency relation information.
'''

import spacy
from spacy import displacy
import pandas as pd

# Set directory info
fpathm = 'C:/Users/vodden/PycharmProjects/TextAnalyticsPipeline/TextAnalyticsPipeline/depparse_visuals/test_files/'
post_file = 'facebook_posts.csv'
depparse_file = 'spacy_depparse.csv'

# Load post data
df_posts = pd.read_csv(fpathm + post_file)
df_posts = df_posts[['platformId', 'message']]

# Load depparse data
df_depparse = pd.read_csv(fpathm + depparse_file)

# Merge df_posts and df_depparse on platformId = identifier
df = df_posts.merge(df_depparse, how='left', left_on='platformId', right_on='identifier')

# Isolate a specific post, this one is about Pauline Hanson
df = df[df['platformId'] == '100063926926757_381321477342080']

# Isolate first sentence
df = df[df['sentence_num'] == 1]

# Sort by word_num
df = df.sort_values(by='word_num')





# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract dependency data from df
selected_columns = ['word_text', 'head_text', 'relation']
df_selected = df[selected_columns]

# Convert the selected DataFrame to a list of dictionaries
dependency_data = df_selected.to_dict(orient='records')
print(dependency_data)

# Create a text string
text = ' '.join(item['word_text'] for item in dependency_data)

# Process the text with spaCy pipeline
doc = nlp(text)

# Set custom extensions
spacy.tokens.Token.set_extension('head', default=None)
spacy.tokens.Token.set_extension('relation', default=None)

# Set custom attributes
for item, token in zip(dependency_data, doc):
    token._.head = item['head_text']
    token._.relation = item['relation']

# Visualize with displacy
displacy.serve(doc, style='dep', auto_select_port=True, options={'compact': True})
