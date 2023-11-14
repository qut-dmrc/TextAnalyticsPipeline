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
from .bigquery_tools import PushTables, Schema #, NLTKSchema, SpaCySchema
from .data_processor import ProcessResults
from .set_up_logging import set_up_logging
from .validate_params import ValidateParams

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

    # Get the processor class and processor name to be used (see config.py)
    pc = ProcessorClass()
    processor_class, processor_name = pc.get_processor_class()

    # Get the language to be used (see config.py)
    lg = Language()
    lang = lg.get_language()

    # Get the library to be used (see config.py)
    lb = Library()
    library = lb.get_library()

    # Set up logging (see set_up_logging.py)
    set_up_logging('TextAnalyticsPipeline/logs', library, processor_name)

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

    # Initialize empty list to store entities dataframes
    result_dfs = []

    if library == 'stanza':

        # Initialize the Stanza model 
        nlp = stanza.Pipeline(f'{lang}', processors=f'tokenize,mwt,{processor_class}', download_method=None)

        if processor_name == 'ner':

                # Set table schema 
                table_schema = Schema.ner_schema

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
                    logging.info('Processing document id: {id}')

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

                        # Run processor
                        entities_df = result_processor.process_ner(id, df)

                        # Append results
                        result_dfs.append([entities_df])

                    else:
                        logging.info('No entities found in document.\n')

        elif processor_name == 'pos':

                # Set table schema 
                table_schema = Schema.pos_schema

                count = 0
                for id, document in zip(identifiers, documents):
                    
                    # Count keeps track of the number of documents processed
                    count = count + 1

                    # Process the document with the Stanza model
                    doc = nlp(document)

                    # Extract the sentences
                    sentences = doc.sentences

                    # Process document part-of-speech tagging
                    logging.info('Processing documents for part-of-speech extraction...')
                    logging.info('Processing document id: {id}')

                    # Initialize result processor
                    result_processor = ProcessResults()

                    data = []
                    for sent_id, sentence in enumerate(doc.sentences, start=1):
                        for word_id, word in enumerate(sentence.words, start=1):
                            data.append([sent_id, word.id, f'{id}_{sent_id}_{word.id}', word.text, word.lemma, word.upos, word.xpos, word.start_char, word.end_char])
                            
                    columns = ['sentence_num', 'word_num', 'word_id', 'word', 'lemma', 'upos', 'xpos', 'start_char', 'end_char']
                    df = pd.DataFrame(data, columns=columns)

                    # Run processor
                    pos_df = result_processor.process_pos(id, df)

                    # Append results
                    result_dfs.append([pos_df])

        elif processor_name == 'depparse':

                # Set table schema 
                table_schema = Schema.depparse_schema

                count = 0
                for id, document in zip(identifiers, documents):
                    
                    # Count keeps track of the number of documents processed
                    count = count + 1

                    # Process the document with the Stanza model
                    doc = nlp(document)

                    # Process depparse
                    logging.info('Processing documents for dependency parsing extraction...')
                    logging.info('Processing document id: {id}')

                    # Initialize result processor
                    result_processor = ProcessResults()

                    dependency_info = []

                    for sentence in doc.sentences:
                        for word in sentence.words:
                            row = {
                                'sentence_num': sentence.index + 1,
                                'word_num': word.id,
                                'word_id': f'{id}_{sentence.index + 1}_{word.id}',
                                'word_text': word.text,
                                'word_lemma': word.lemma,
                                'word_start_char': word.start_char,
                                'word_end_char': word.end_char
                            }

                            if word.deprel == 'root':
                                relation = word.deprel.upper()
                                row.update({
                                    'relation': relation,
                                    'head_num': word.id,
                                    'head_id': f'{id}_{sentence.index + 1}_{word.id}',
                                    'head_text': word.text,
                                    'head_lemma': word.lemma,
                                    'head_start_char': word.start_char,
                                    'head_end_char': word.end_char
                                })
                            else:
                                head_word = sentence.words[int(word.head) - 1]
                                head_info = {
                                    'relation': word.deprel,
                                    'head_num': word.head,
                                    'head_id': f'{id}_{sentence.index + 1}_{word.head}',
                                    'head_text': head_word.text,
                                    'head_lemma': head_word.lemma,
                                    'head_start_char': head_word.start_char,
                                    'head_end_char': head_word.end_char
                                }
                                row.update(head_info)

                            dependency_info.append(row)

                    # Transform in df
                    df = pd.DataFrame(dependency_info)

                    # Run processor
                    depparse_df = result_processor.process_depparse(id, df)

                    # Append results
                    result_dfs.append([depparse_df])

        else:
            result_dfs = None

    elif library == 'spacy':
        
        # Initialize the Spacy model 
        nlp = spacy.load(f'{lang}_core_web_lg')
        #TODO deal with sizes of language models 
        
        if processor_name == 'ner':

                # Set table schema 
                table_schema = Schema.ner_schema

                count = 0
                for id, document in zip(identifiers, documents):

                    # Count keeps track of the number of documents processed
                    count = count + 1

                    # Process the document with the Spacy model
                    doc = nlp(document)

                    print(
                        f'Document ID: {id}\n',
                        f'Document Text: {document}'
                    )
                    
                    # Extract the entities
                    entities = doc.ents

                    # Process document entities
                    logging.info('Processing documents for entity extraction...')
                    logging.info('Processing document id: {id}')

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

                        # Run processor
                        entities_df = result_processor.process_ner(id, df)

                        # Append results
                        result_dfs.append([entities_df])

                    else:
                        logging.info('No entities found in document.\n')

        elif processor_name == 'pos':
            
            # Set table schema 
            table_schema = Schema.pos_schema

            count = 0
            for id, document in zip(identifiers, documents):
                
                # Count keeps track of the number of documents processed
                count = count + 1

                # Process the document with the Spacy model
                doc = nlp(document)

                # Process document part-of-speech tagging
                logging.info('Processing documents for part-of-speech extraction...')
                logging.info('Processing document id: {id}')

                # Initialize result processor
                result_processor = ProcessResults()

                # Create empty list to hold tokens dictionaries
                tokens_list = []

                # Initialize sentence number
                sentence_num = 1

                # Get tokens information
                for sentence in doc.sents:
                    word_num = 1  # Initialize word_num for each sentence
                    for token in sentence:
                        tokens_list.append({
                            'sentence_num': sentence_num,
                            'word_num': word_num,
                            'word_id': f'{id}_{sentence_num}_{word_num}',
                            'word': token.text,
                            'lemma': token.lemma_,
                            'upos': token.pos_,
                            'xpos': token.tag_,
                            'start_char': token.idx,
                            'end_char': token.idx + len(token)
                        })
                        word_num += 1  # Increment word_num for each word in the sentence
                    sentence_num += 1  # Increment sentence_num for each sentence in the document

                # Convert tokens_list = to dataframe
                df = pd.DataFrame(tokens_list)

                # Run processor
                pos_df = result_processor.process_pos(id, df)

                # Append results
                result_dfs.append([pos_df])

        elif processor_name == 'depparse':

            # Set table schema 
            table_schema = Schema.depparse_schema

            count = 0
            for id, document in zip(identifiers, documents):
                
                # Count keeps track of the number of documents processed
                count = count + 1

                # Process the document with the Spacy model
                doc = nlp(document)

                # Process depparse
                logging.info('Processing documents for dependency parsing extraction...')
                logging.info('Processing document id: {id}')

                # Initialize result processor
                result_processor = ProcessResults()

                # Initialize sentence number
                sentence_num = 1

                # Get information
                dependency_info = []
                for sentence in doc.sents:

                    word_counter = 1 
                    word_positions = {}  # Dictionary to store word positions within each sentence
                    for token in sentence:
                        word_positions[token.i] = word_counter
                        word_counter += 1

                    for token in sentence:
                        head_token = token.head
                        dependency_info.append({
                            'sentence_num': sentence_num,
                            'word_num': word_positions[token.i],
                            'word_id': f'{id}_{sentence_num}_{word_positions[token.i]}',
                            'word_text': token.text,
                            'word_lemma': token.lemma_,
                            'start_char': token.idx,
                            'end_char': token.idx + len(token),
                            'relation': token.dep_,
                            'head_num': word_positions[head_token.i] if head_token.i in word_positions else 0,
                            'head_id': f'{id}_{sentence_num}_{word_positions[head_token.i] if head_token.i in word_positions else 0}',
                            'head_text': head_token.text,
                            'head_lemma': head_token.lemma_,
                            'head_start_char': head_token.idx,
                            'head_end_char': head_token.idx + len(head_token)
                        })
                    sentence_num += 1  # Increment sentence_num for each sentence in the document

                # Transform in df
                df = pd.DataFrame(dependency_info)

                # Run processor
                depparse_df = result_processor.process_depparse(id, df)

                # Append results
                result_dfs.append([depparse_df])

    # Specify a chunk size so that when result_dfs reaches chunk size, it is pushed to BigQuery
    chunk = find_optimal_chunk_size(n_docs)

    # If result_dfs > 5000 rows, OR if count(docs processed) is greater than the total n_docs - chunk (i.e. left over after chunks processed), push to BigQuery (remainders will be pushed one doc at a time)
    if len(result_dfs) == chunk or count > n_docs - chunk:

        # Concatenate dfs by getting the first item from each list in result_dfs
        dfs_concat = [inner_list[0] for inner_list in result_dfs]
        results_dfs_concat = pd.concat(dfs_concat, ignore_index=True)

        # Write results_dfs_concat to csv
        logging.info(f'Writing {processor_name} to csv...')
        results_dfs_concat.to_csv(f'TextAnalyticsPipeline/temp/temp_{processor_name}_{library}.csv', encoding='utf-8', index=False)

        # Push results_dfs_concat to BigQuery
        push_tables = PushTables()
        logging.info(f'Pushing {processor_name} to BigQuery...')
        push_tables.push_to_gbq(database_import, bq, project, dataset, table, table_schema, library, proc=f'{processor_name}')
        logging.info(f'Results of {processor_name} successfully pushed to BigQuery!\n')

        # empty result_dfs list
        result_dfs = []

    exit()
