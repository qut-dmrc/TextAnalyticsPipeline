'''
Uses Stanza, Spacy and NLTK to process different text analysis tasks

Documentation: 
https://stanfordnlp.github.io/stanza/
https://spacy.io/
https://www.nltk.org/

'''

# Import Statements
import stanza
import spacy

import os
from os.path import exists
import logging
import glob
import pandas as pd

from google.cloud import bigquery
from google.cloud.bigquery.client import Client
from google.api_core import exceptions

# local imports
from .config import BigQuery, InputConf, ProcessorClass, Language, Library
from .bigquery_tools import PushTables, StanzaSchema #, NLTKSchema, SpaCySchema
from .data_processor import ProcessResults
from .set_up_logging import set_up_logging
from .validate_params import ValidateParams

# stanza.download('en') # download English model

# Google BigQuery Config
# ----------------------------------------------------------------------------------------------------------------------
def find_optimal_chunk_size(a, min_chunk_size=5000, max_chunk_size=10000):
    # Start with the maximum allowed chunk size
    chunk_size = max_chunk_size

    # Iterate from the maximum allowed chunk size down to the minimum
    while chunk_size >= min_chunk_size:
        # Check if the dataframe length can be evenly divided by the current chunk size
        if a % chunk_size == 0:
            return chunk_size
        chunk_size -= 1

    # If no suitable chunk size is found, set it to min_chunk_size
    return min_chunk_size


def run_text_pipeline():
    '''
    Runs the text analysis pipeline; the starting point for the pipeline.
    '''

    # Set up logging (see set_up_logging.py)
    set_up_logging('TextAnalyticsPipeline/logs')

    # Initialize config classes
    gbq = BigQuery()
    inp = InputConf()

    # Get config details from config classes
    project = gbq.project_name
    dataset = gbq.dataset_name
    table = gbq.tablename
    id_column = inp.id_column
    text_column = inp.text_column
    database_import = inp.from_database

    # Get GBQ credentials from environment variables, or from local file in 'access_key' directory. If multiple keys exist, match with project id
    try:
        gbq_creds = os.environ['gbq_servicekey']
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gbq_creds
    except:
        # Check for Google access key; looks for .json service account key in the 'access_key' dir
        servicekeypath = glob.glob(f'{os.getcwd()}/TextAnalyticsPipeline/access_key/*.json')

        if len(servicekeypath) > 0:
            for potential_key in servicekeypath:
                with open(potential_key, 'r') as f:
                    contents = f.read()
                    if f'"project_id": "{project}"' in contents:
                        gbq_creds = potential_key
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gbq_creds
                        break
                    # If no matching key found, exit
                    else:
                        logging.info('No compatible Google BigQuery credentials found in /access_key. Exiting.')
                        exit()
        else:
            logging.info('No Google BigQuery credentials found in /access_key. Exiting.')
            exit()

    # Initialise BigQuery client
    client = bigquery.Client()
    bq = Client(project=project)

    # If database_import is True, query BigQuery table. If False, read csv into dataframe
    if database_import == True:
        # Prepare SQL query
        query_string = f"""
            SELECT DISTINCT
            {id_column}, {text_column}
            FROM `{project}.{dataset}.{table}`
            """

        # Validate input parameters
        vdp = ValidateParams()
        project, dataset, table = vdp.validate_project_parameters(project, dataset, table, bq)

        # Run query and save to dataframe
        try:
            logging.info(f"Querying '{table}' table...")
            df = (bq.query(query_string).result().to_dataframe()).dropna(subset=text_column)
            n_docs = len(df)
            logging.info(query_string)
            logging.info("Query successful\n")
        except exceptions.NotFound:
            logging.info(f"Table '{table}' not found in dataset '{dataset}'. Exiting.")
            exit()

    else:
        # Read csv into dataframe
        csv_path = './TextAnalyticsPipeline/input_csv/'
        # glob to locate csv file
        input_csv = glob.glob(csv_path + '*.csv')[0]
        if len(input_csv) > 0:
            try:
                df = pd.read_csv(input_csv)
                logging.info(f"Loaded '{input_csv}' from input directory.\n")
            except FileNotFoundError:
                logging.info(f"File '{input_csv}' not found. Exiting.")
                exit()
        else:
            logging.info(f"No csv files found in '{csv_path}'. Exiting.")
            exit()

    # Get identifiers and documents from dataframe
    identifiers = df[id_column].tolist()
    documents = df[text_column].tolist()

    # Get the processor class to be used based on the config
    pc = ProcessorClass()
    processor_class = pc.get_processor_class()

    # Get the language
    lg = Language()
    lang = lg.get_language()

    # Get the library to be used based on the config
    lb = Library()
    library = lb.get_library()

    # Initialize empty list to store entities dataframes
    result_dfs = []

    if library == 'stanza':

        # Initialize the Stanza model 
        nlp = stanza.Pipeline(f'{lang}', processors=f'tokenize,mwt,{processor_class}', download_method=None)

        if processor_class == 'ner':

                # Set table schema 
                table_schema = StanzaSchema.ner_schema

                count = 0
                for id, document in zip(identifiers, documents):

                    # Count keeps track of the number of documents processed
                    count = count + 1

                    # Process the document with the Stanza model
                    doc = nlp(document)

                    print(
                        f'Document ID: {id}\n',
                        f'Document Text: {document}'
                    )
                    
                    # Extract the entities
                    entities = doc.entities

                    # Process document entities
                    logging.info('Processing documents for entity extraction...')

                    # Test if there is entities, and, if so, process them
                    if len(entities) > 0:

                        # Convert entities to dataframe and extract dictionary values
                        df = pd.DataFrame(entities)
                        
                        # Transforming the stanza output to a dict
                        df[0] = df[0].apply(lambda x: x.to_dict())

                        # Extract values from dictionary
                        df = pd.json_normalize(df[0])

                        # Initialize result processor
                        result_processor = ProcessResults()
                        entities_df = result_processor.process_ner(id, df)
                        result_dfs.append([entities_df])

                    else:
                        logging.info('No entities found in document.\n')

        elif processor_class == 'lemma, pos, depparse':

                count = 0
                for id, document in zip(identifiers, documents):

                    # Count keeps track of the number of documents processed
                    count = count + 1

                    # Process the document with the Stanza model
                    doc = nlp(document)

                    sentences = doc.sentences
                    table_schema = [StanzaSchema.sentences_schema, StanzaSchema.dependencies_schema]
                    # Process document sentences
                    logging.info('Processing sentences for part-of-speech extraction...')
                    sentences_df, dependencies_df = result_processor.process_sentences(id, sentences, count)
                    result_dfs.append([sentences_df, dependencies_df])

        else:
            result_dfs = None

        # Specify a chunk size so that when result_dfs reaches chunk size, it is pushed to BigQuery
        chunk = find_optimal_chunk_size(n_docs)

        # If result_dfs > 5000 rows, OR if count(docs processed) is greater than the total n_docs - chunk (i.e. left over after chunks processed), push to BigQuery (remainders will be pushed one doc at a time)
        if len(result_dfs) == chunk or count > n_docs - chunk:

            # If len(result_dfs[0] == 1, this is NER. Concatenate result_dfs[0]
            if len(result_dfs[0]) == 1:

                # Concatenate entities dfs by getting the first item from each list in result_dfs
                dfs_concat = [inner_list[0] for inner_list in result_dfs]
                entities_dfs_concat = pd.concat(dfs_concat, ignore_index=True)

                # Write entities_df to csv
                logging.info('Writing entities to csv...')
                entities_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_entities_stanza.csv', encoding='utf-8', index=False)

                # Push entities to BigQuery
                push_tables = PushTables()
                logging.info('Pushing entities to BigQuery...')
                push_tables.push_to_gbq(database_import, bq, project, dataset, table, table_schema, library, proc='entities')
                logging.info('Entities successfully pushed to BigQuery!\n')
                # empty result_dfs list
                result_dfs = []

            # But if len(result_dfs[0]) > 1, this is POS, DEPPARSE. Concatenate all dfs in result_dfs[0]
            else:
                # Concatenate sentences dfs
                result_dfs_sent = []
                for i in range(len(result_dfs)):
                    result_dfs_sent.append(result_dfs[i][0])
                sentences_dfs_concat = pd.concat(result_dfs_sent, axis=0)

                # Column order
                # Set sentences_df column order as Schema.sentences_column_order. If column not in df, add it and set value to nan
                sentences_column_order = StanzaSchema.sentences_column_order
                for column in sentences_column_order:
                    if column not in sentences_dfs_concat.columns:
                        sentences_dfs_concat[column] = pd.NA
                sentences_dfs_concat = sentences_dfs_concat[sentences_column_order]

                # Write sentences_df to csv
                logging.info('Writing sentences to csv...')
                sentences_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_sentences_stanza.csv', encoding='utf-8', index=False)
                records = len(sentences_dfs_concat)
                # Push sentences to BigQuery
                push_tables = PushTables()
                logging.info('Pushing sentences to BigQuery...')
                push_tables.push_to_gbq(database_import, bq, project, dataset, table, records, table_schema=StanzaSchema.sentences_schema, proc='sentences')
                logging.info('Sentences successfully pushed to BigQuery!\n')

                # Concatenate dependencies dfs
                result_dfs_depparse = []
                for i in range(len(result_dfs)):
                    result_dfs_depparse.append(result_dfs[i][1])
                dependencies_dfs_concat = pd.concat(result_dfs_depparse, axis=0)

                # Set dependencies_df column order as Schema.dependencies_column_order. If column not in df, add it and set value to nan
                dependencies_column_order = StanzaSchema.dependencies_column_order
                for column in dependencies_column_order:
                    if column not in dependencies_dfs_concat.columns:
                        dependencies_dfs_concat[column] = pd.NA

                dependencies_dfs_concat = dependencies_dfs_concat[dependencies_column_order]
                # Write dependencies_df to csv
                logging.info('Writing dependencies to csv...')
                dependencies_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_depparse_stanza.csv', encoding='utf-8', index=False)
                records = len(dependencies_dfs_concat)
                # Push dependencies to BigQuery
                push_tables = PushTables()
                logging.info('Pushing dependencies to BigQuery...')
                push_tables.push_to_gbq(database_import, bq, project, dataset, table, records, table_schema=StanzaSchema.dependencies_schema, proc='depparse')
                logging.info('Dependencies successfully pushed to BigQuery!\n')
                # empty result_dfs list
                result_dfs = []

    elif library == 'spacy':
        
        # Initialize the Spacy model 
        nlp = spacy.load(f'{lang}_core_web_lg')
        

        if processor_class == 'ner':

                # Set table schema 
                table_schema = StanzaSchema.ner_schema

                count = 0
                for id, document in zip(identifiers, documents):

                    # Count keeps track of the number of documents processed
                    count = count + 1

                    # Process the document with the Stanza model
                    doc = nlp(document)

                    print(
                        f'Document ID: {id}\n',
                        f'Document Text: {document}'
                    )
                    
                    # Extract the entities
                    entities = doc.ents

                    # Process document entities
                    logging.info('Processing documents for entity extraction...')

                    # Test if there is entities, and, if so, process them
                    if len(entities) > 0:

                        # Create empty list to hold entities dictionaries
                        entities_list = []

                        # Get entities information
                        for ent in entities:
                            entities_list.append({
                                'text': ent.text,
                                'type': ent.label_,
                                'start_char': ent.start_char,
                                'end_char': ent.end_char,
                            })

                        # Convert entities list to dataframe
                        df = pd.DataFrame(entities_list)

                        # Initialize result processor
                        result_processor = ProcessResults()
                        entities_df = result_processor.process_ner(id, df)
                        result_dfs.append([entities_df])

                    else:
                        logging.info('No entities found in document.\n')


        elif processor_class == 'lemma, pos':
            
            # Load the Spacy model (outside the loop to improve performance)
            nlp = spacy.load(f'{lang}_core_web_lg')

        
        # Specify a chunk size so that when result_dfs reaches chunk size, it is pushed to BigQuery
        chunk = find_optimal_chunk_size(n_docs)

        # If result_dfs > 5000 rows, OR if count(docs processed) is greater than the total n_docs - chunk (i.e. left over after chunks processed), push to BigQuery (remainders will be pushed one doc at a time)
        if len(result_dfs) == chunk or count > n_docs - chunk:

            # If len(result_dfs[0] == 1, this is NER. Concatenate result_dfs[0]
            if len(result_dfs[0]) == 1:

                # Concatenate entities dfs by getting the first item from each list in result_dfs
                dfs_concat = [inner_list[0] for inner_list in result_dfs]
                entities_dfs_concat = pd.concat(dfs_concat, ignore_index=True)

                # Write entities_df to csv
                logging.info('Writing entities to csv...')
                entities_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_entities_stanza.csv', encoding='utf-8', index=False)

                # Push entities to BigQuery
                push_tables = PushTables()
                logging.info('Pushing entities to BigQuery...')
                push_tables.push_to_gbq(database_import, bq, project, dataset, table, table_schema, library, proc='entities')
                logging.info('Entities successfully pushed to BigQuery!\n')
                # empty result_dfs list
                result_dfs = []

            # But if len(result_dfs[0]) > 1, this is POS, DEPPARSE. Concatenate all dfs in result_dfs[0]
            else:
                # Concatenate sentences dfs
                result_dfs_sent = []
                for i in range(len(result_dfs)):
                    result_dfs_sent.append(result_dfs[i][0])
                sentences_dfs_concat = pd.concat(result_dfs_sent, axis=0)

                # Column order
                # Set sentences_df column order as Schema.sentences_column_order. If column not in df, add it and set value to nan
                sentences_column_order = StanzaSchema.sentences_column_order
                for column in sentences_column_order:
                    if column not in sentences_dfs_concat.columns:
                        sentences_dfs_concat[column] = pd.NA
                sentences_dfs_concat = sentences_dfs_concat[sentences_column_order]

                # Write sentences_df to csv
                logging.info('Writing sentences to csv...')
                sentences_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_sentences_stanza.csv', encoding='utf-8', index=False)
                records = len(sentences_dfs_concat)
                # Push sentences to BigQuery
                push_tables = PushTables()
                logging.info('Pushing sentences to BigQuery...')
                push_tables.push_to_gbq(database_import, bq, project, dataset, table, records, table_schema=StanzaSchema.sentences_schema, proc='sentences')
                logging.info('Sentences successfully pushed to BigQuery!\n')

                # Concatenate dependencies dfs
                result_dfs_depparse = []
                for i in range(len(result_dfs)):
                    result_dfs_depparse.append(result_dfs[i][1])
                dependencies_dfs_concat = pd.concat(result_dfs_depparse, axis=0)

                # Set dependencies_df column order as Schema.dependencies_column_order. If column not in df, add it and set value to nan
                dependencies_column_order = StanzaSchema.dependencies_column_order
                for column in dependencies_column_order:
                    if column not in dependencies_dfs_concat.columns:
                        dependencies_dfs_concat[column] = pd.NA

                dependencies_dfs_concat = dependencies_dfs_concat[dependencies_column_order]
                # Write dependencies_df to csv
                logging.info('Writing dependencies to csv...')
                dependencies_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_depparse_stanza.csv', encoding='utf-8', index=False)
                records = len(dependencies_dfs_concat)
                # Push dependencies to BigQuery
                push_tables = PushTables()
                logging.info('Pushing dependencies to BigQuery...')
                push_tables.push_to_gbq(database_import, bq, project, dataset, table, records, table_schema=StanzaSchema.dependencies_schema, proc='depparse')
                logging.info('Dependencies successfully pushed to BigQuery!\n')
                # empty result_dfs list
                result_dfs = []

    exit()
