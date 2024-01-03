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
    
    def process_morphology(self, id, df):
        '''
        For Morphology: Processes the results into a table with the following columns:
            'identifier',
            'sentence_num',
            'word_num', 
            'word_id',
            'word',
            'lemma',
            'features_Number',
            'features_Mood',
            'features_Person',
            'features_Tense',
            'features_VerbForm',
            'features_Case',
            'features_Gender',
            'features_PronType',
            'features_Degree',
            'features_Definite',
            'features_NumForm',
            'features_NumType',
            'features_Voice',
            'start_char',
            'end_char'
        '''

        # Concatenate id column values and entities_df as new dataframe
        id_list = [id] * len(df)
        morphology_df = pd.concat([pd.DataFrame(id_list), df], axis=1)
        morphology_df.columns = ['identifier',
            'sentence_num',
            'word_num', 
            'word_id',
            'word',
            'lemma',
            'features_Number',
            'features_Mood',
            'features_Person',
            'features_Tense',
            'features_VerbForm',
            'features_Case',
            'features_Gender',
            'features_PronType',
            'features_Degree',
            'features_Definite',
            'features_NumForm',
            'features_NumType',
            'features_Voice',
            'start_char',
            'end_char']

        return morphology_df

