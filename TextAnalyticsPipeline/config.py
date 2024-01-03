import os 
import yaml
import glob

# Get the current working directory
wd = os.getcwd()

# Attempt to open and load the YAML configuration file
try:
    with open(f'{wd}/TextAnalyticsPipeline/config/config.yml', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

# Handle FileNotFoundError if the specified config file is not found
except FileNotFoundError:
    print("\nCannot find config file!")
    exit()

# Handle yaml.YAMLError if there's an issue with the YAML format in the config file
except yaml.YAMLError:
    print("\nDetected an issue with your config.")


class BigQuery:
    try:
        gbq_creds = os.environ['gbq_servicekey']
    except KeyError:
        pass

    project_name = config['project_name']
    dataset_name = config['dataset_name']
    tablename = config['tablename']

class InputConf:
    from_database = config['from_database']
    from_csv = config['from_csv']
    id_column = config['id_column']
    text_column = config['text_column']

class ProcessorClass:

    def get_processor_class(self):

        ner = config['named_entity_recognition']
        pos = config['part_of_speech']
        depparse = config['dependency_parsing']
        sentiment = config['sentiment']
        morphology = config['morphology']

        # If any of ner, pos, depparse, or sentiment are True, then set processor_class to whichever is True
        if ner == True:
            processor_class = 'ner'
            processor_name = 'ner'
        elif pos == True:
            processor_class = 'lemma, pos'
            processor_name = 'pos'
        elif depparse == True:
            processor_class = 'lemma, pos, depparse'
            processor_name = 'depparse'
        elif sentiment == True:
            processor_class = 'sentiment'
            processor_name = 'sentiment'
        elif morphology == True:
            processor_class = 'lemma, pos'
            processor_name = 'morphology'
        else:
            processor_class = None
            processor_name = None

        # If more than once of processor_class is True, then exit
        if [ner, pos, depparse, sentiment].count(True) > 1:
            print("\nMore than one processor class is True. Please set only one to True.")
            exit()
        else:
            pass

        return processor_class, processor_name
    
class Language:

    def get_language(self):

        lang = config['language']
        return lang

class Library:

    def get_library(self):

        l_stanza = config['stanza']
        l_spacy = config['spacy']
        l_nltk = config['nltk']
    
        # If any of ner, pos, or sentiment are True, then set processor_class to whichever is True
        if l_stanza == True:
            library = 'stanza'
        elif l_spacy == True:
            library = 'spacy'
        elif l_nltk == True:
            library = 'nltk'
        else:
            library = None

        # If more than once of processor_class is True, then exit
        if [l_stanza, l_spacy, l_nltk].count(True) > 1:
            print("\nMore than one library is True. Please set only one to True.")
            exit()
        else:
            pass

        return library
