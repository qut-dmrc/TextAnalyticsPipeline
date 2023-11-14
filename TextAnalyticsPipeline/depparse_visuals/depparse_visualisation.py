'''
Takes a dependency parsing table and generates a dependency tree from it.
'''

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

# test_sentence = "In case you haven't already heard the news, Pauline is coming to WA!"
# # It should look like this:
# import spacy
# from spacy import displacy
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(test_sentence)
# displacy.serve(doc, style="dep", options={'compact': True})
# print('http://localhost:5000')


# Now, I want to replicate this manually using the depparse table

# First, I need to get the root of the sentence
root = df[df['relation'] == 'ROOT']
root = root['word_text'].values[0]

# Next, I need to get the children of the root
children_levels = []
all_children = []

# Now, I need to iteratively get the children of the children until there are no more children
current_level = [root]
level_number = 0
visited = set()

while current_level:
    next_level = []
    for child in current_level:
        if child not in visited:
            children = df[df['head_text'] == child]
            next_level.extend(children['word_text'].tolist())
            visited.add(child)
            if len(children) > 0:
                all_children.append(children)     # Append the DataFrame for the current level

    if next_level:
        children_levels.append(next_level)

    current_level = next_level
    level_number += 1

print('done!')

# Now, children_levels is a list of lists where each sublist represents children at a different level
# all_children is a list of DataFrames, each representing the children at a different level

#children_levels[0] are the children of the root
#children_levels[1] are the children of the children of the root


# The next step is to



