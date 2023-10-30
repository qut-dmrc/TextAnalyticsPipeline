'''
Contians functions relating to the cleaning of pulled data.
'''
import pandas as pd

class ProcessResults:

    def process_ner(self, id, df):
        '''
        For Named Entity Recognition: Processes the results into a table with the following columns:
            - identifier
            - text
            - type
            - start_char
            - end_char
        '''
        
        # Concatenate id column values and entities_df as new dataframe
        id_list = [id] * len(df)
        entities_df = pd.concat([pd.DataFrame(id_list), df], axis=1)
        entities_df.columns = ['identifier', 'text', 'type', 'start_char', 'end_char']

        return entities_df
    
    def process_pos(self, id, df):
        '''
        For Part of Speech: Processes the results into a table with the following columns:
            'identifier',
            'sentence_num',
            'word_num',
            'word_id',
            'word',
            'lemma',
            'upos',
            'xpos',
            'start_char',
            'end_char',
        '''
        # Concatenate id column values and entities_df as new dataframe
        id_list = [id] * len(df)
        pos_df = pd.concat([pd.DataFrame(id_list), df], axis=1)
        pos_df.columns = ['identifier', 'sentence_num', 'word_num', 'word_id', 'word', 'lemma', 'upos', 'xpos', 'start_char', 'end_char']

        return pos_df

    def process_depparse(self, id, df):
        '''
        For Dependency Parsing: Processes the results into a table with the following columns:
            'identifier',
            'sentence_num',
            'word_num', 
            'word_id',
            'word_text',
            'word_lemma',
            'word_start_char',
            'word_end_char',
            'relation',
            'head_num',
            'head_id',
            'head_text',
            'head_lemma',
            'head_start_char',
            'head_end_char'
        '''

        # Concatenate id column values and entities_df as new dataframe
        id_list = [id] * len(df)
        depparse_df = pd.concat([pd.DataFrame(id_list), df], axis=1)
        depparse_df.columns = ['identifier', 'sentence_num', 'word_num', 'word_id', 'word_text', 'word_lemma', 'word_start_char', 'word_end_char', 'relation', 'head_num', 'head_id', 'head_text', 'head_lemma', 'head_start_char', 'head_end_char']

        return depparse_df
    
    def process_sentences(self, id, sentences, count):

        '''
        For Part-of-Speech Tagging, includes dependency parsing
        '''
        print(count)
        # Convert sentences to dataframe and extract dictionary values
        sent_df = pd.DataFrame(sentences)
        sent_df[0] = sent_df[0].apply(lambda x: x.to_dict())
        sent_df = pd.json_normalize(sent_df[0])
        # stack columns
        sent_df_stack = sent_df.stack().reset_index().rename(columns={'level_0': 'sentence_num', 'level_1': 'word_num', 0: 'sentence_value'})
        # Extract values from dictionaries
        sent_df_unstack = pd.json_normalize(sent_df_stack['sentence_value'])

        # Concatenate id column values and sent_df_unstack as new dataframe
        id_list = [id] * len(sent_df_unstack)
        sentences_df = pd.concat([pd.DataFrame(id_list), sent_df_stack['sentence_num'], sent_df_unstack], axis=1)
        # sentences_df.columns = sentences_df.columns = ['identifier', 'sentence_num', 'word_num', 'word', 'lemma', 'upos', 'xpos', 'head', 'deprel', 'start_char', 'end_char', 'features']
        sentences_df = sentences_df.rename(columns={0:'identifier', 'id':'word_num', 'text':'word', 'feats':'features'})
        sentences_df['word_id'] = sentences_df['identifier'].astype(str) + '_' + sentences_df['sentence_num'].astype(str) + '_' + sentences_df['word_num'].astype(str)

        # Extract features from dictionary
        if 'features' in sentences_df.columns:
            # Extract features from dictionary
            features_df = sentences_df[['word_id', 'features']]
            # Convert features_df['features'] to dictionaries
            features_df = features_df.dropna(subset=['features'])
            features_df['features'] = features_df['features'].apply(lambda x: dict(item.split("=") for item in x.split("|")))
            features_unpack = pd.json_normalize(features_df['features']).add_prefix('features_')

            # Concatenate word_id column values and features_unpack as new dataframe
            features_df = pd.concat([features_df['word_id'], features_unpack], axis=1)
        else:
            features_df = pd.DataFrame()
            # TODO get all cols
            features_df[['word_id', 'features_Case', 'features_Gender', 'features_Number',
                                   'features_Person', 'features_PronType', 'features_Mood',
                                   'features_Tense', 'features_VerbForm', 'features_Degree',
                                   'features_Definite']] = pd.NA
        sentences_df = pd.merge(sentences_df, features_df, on='word_id', how='left')


        # drop 'features' column from sentences_df
        if 'features' in sentences_df.columns:
            sentences_df = sentences_df.drop(columns=['features'])

        # Extract dependencies from dictionary
        # for each sentence in sentences list, extract dependencies
        dependencies_df = pd.DataFrame()
        unpacked_data = []
        for sentence in sentences:
            dependencies = sentence.dependencies

            for item in dependencies:

                word_dict = item[0]
                relation = item[1]
                head_dict = item[2]

                unpacked_data.append({
                    'sentence_num': sentence.sent_id,
                    'word_num': word_dict.id,
                    'word_text': word_dict.text,
                    'word_lemma': word_dict.lemma,
                    'relation': relation,
                    'head_id': head_dict.id,
                    'head_text': head_dict.text,
                    'head_lemma': head_dict.lemma,
                    'head_upos': head_dict.upos,
                    'head_xpos': head_dict.xpos,
                    'head_feats': head_dict.feats,
                    'head_head': head_dict.head,
                    'head_deprel': head_dict.deprel,
                    'head_start_char': head_dict.start_char,
                    'head_end_char': head_dict.end_char
                })

        dependencies_df = pd.DataFrame(unpacked_data)

        # extract features from target_feats column and add to dependencies_df, ignore 'None' values
        dependencies_df['identifier'] = id
        dependencies_df['head_feats'] = dependencies_df['head_feats'].apply(lambda x: dict(item.split("=") for item in x.split("|")) if x != None else None)
        dependencies_df = pd.concat([dependencies_df, pd.json_normalize(dependencies_df['head_feats']).add_prefix('head_feats_')], axis=1)
        dependencies_df['word_id'] = dependencies_df['identifier'].astype(str) + '_' + dependencies_df['sentence_num'].astype(str) + '_' + dependencies_df['word_num'].astype(str)
        dependencies_df = dependencies_df.drop(columns=['head_feats'])

        return sentences_df, dependencies_df
