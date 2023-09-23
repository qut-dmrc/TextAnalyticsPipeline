import os
import yaml
import glob

wd = os.getcwd()

try:
    with open(f'{wd}/TextAnalyticsPipeline/config/config.yml', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    print("\nCannot find config file!")
    exit()
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
        sentiment = config['sentiment']
        lang = config['language']

        # If any of ner, pos, or sentiment are True, then set processor_class to whichever is True
        if ner == True:
            processor_class = 'ner'
        elif pos == True:
            processor_class = 'lemma, pos, depparse'
        elif sentiment == True:
            processor_class = 'sentiment'
        else:
            processor_class = None

        # If more than once of processor_class is True, then exit
        if [ner, pos, sentiment].count(True) > 1:
            print("\nMore than one processor class is True. Please set only one to True.")
            exit()
        else:
            pass

        if lang == 'en':
            mwt = ''
        else:
            mwt = 'mwt,'

        return processor_class, mwt, lang

