'''
This file contains the Google BigQuery table schema for the entity extraction output table, and the function to push the
output to BigQuery.
'''

import os
import glob
import pandas as pd

from google.cloud import bigquery
from google.cloud.bigquery.client import Client
from google.cloud.exceptions import NotFound

from .set_up_logging import *


class GBQCreds:

    def get_gbq_creds(self, project):
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


    def get_client(self, project):
        client = bigquery.Client()
        bq = Client(project=project)

        return bq

class GetCSVFiles:

    def read_csv_from_file(self, logging):
        # Read csv from /input_csv/ directory into dataframe
        csv_path = './TextAnalyticsPipeline/input_csv/'
        # Glob to locate csv file
        input_csv = glob.glob(csv_path + '*.csv')[0]  # Set to [0] for now, TODO accept multiple input files
        if len(input_csv) > 0:
            try:
                df = pd.read_csv(input_csv)
                n_docs = len(df)
                logging.info(f"Loaded '{input_csv}' from input directory.\n")
            except FileNotFoundError:
                logging.info(f"File '{input_csv}' not found. Exiting.")
                exit()
        else:
            logging.info(f"No csv files found in '{csv_path}'. Exiting.")
            exit()

        return n_docs, df

class Schema:

    # Named Entity Recognition schema
    ner_schema = [
        bigquery.SchemaField("identifier", "STRING", mode="REQUIRED", description="Identifier for the record"),
        bigquery.SchemaField("text", "STRING", mode="NULLABLE", description="Named entity text"),
        bigquery.SchemaField("type", "STRING", mode="NULLABLE", description="Named entity type"),
        bigquery.SchemaField("start_char", "INTEGER", mode="NULLABLE", description="Location of start character of entity in input string."),
        bigquery.SchemaField("end_char", "INTEGER", mode="NULLABLE", description="Location of end character of entity in input string.")
    ]

    # Column order for Named Entity Recognition table
    ner_column_order = [
        'identifier',
        'text',
        'type',
        'start_char',
        'end_char'
    ]

    # Part of Speech Tagging schema
    pos_schema = [
        bigquery.SchemaField('identifier', 'STRING', description='Identifier for the record'),
        bigquery.SchemaField('sentence_num', 'INTEGER', description='Sentence number'),
        bigquery.SchemaField('word_num', 'INTEGER', description='Word number in the sentence'),
        bigquery.SchemaField('word_id', 'STRING', description='Word identifier'),
        bigquery.SchemaField('word', 'STRING', description='Word text'),
        bigquery.SchemaField('lemma', 'STRING', description='Lemma of the word'),
        bigquery.SchemaField('upos', 'STRING', description='Universal Part-of-Speech tag'),
        bigquery.SchemaField('xpos', 'STRING', description='Language-specific Part-of-Speech tag'),
        bigquery.SchemaField('start_char', 'INTEGER', description='Start character position in text'),
        bigquery.SchemaField('end_char', 'INTEGER', description='End character position in text'),
    ]

    # Column order for Part of Speech Tagging table
    pos_column_order = [
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
        ]

    # Dependency Parsing schema
    depparse_schema = [
        bigquery.SchemaField('identifier', 'STRING', description='Identifier for the record'),
        bigquery.SchemaField('sentence_num', 'INTEGER', description='Sentence number'),
        bigquery.SchemaField('word_num', 'INTEGER', description='Source word identifier'),
        bigquery.SchemaField('word_id', 'STRING', description='Source word identifier'),
        bigquery.SchemaField('word_text', 'STRING', description='Source word text'),
        bigquery.SchemaField('word_lemma', 'STRING', description='Source word lemma'),
        bigquery.SchemaField('word_start_char', 'INTEGER', description='Start character position in text'),
        bigquery.SchemaField('word_end_char', 'INTEGER', description='End character position in text'),
        bigquery.SchemaField('relation', 'STRING', description='Dependency relation'),
        bigquery.SchemaField('head_num', 'STRING', description='Target word identifier'),
        bigquery.SchemaField('head_id', 'STRING', description='Target word identifier'),
        bigquery.SchemaField('head_text', 'STRING', description='Target word text'),
        bigquery.SchemaField('head_lemma', 'STRING', description='Target word lemma'),
        bigquery.SchemaField('head_start_char', 'INTEGER', description='Target word start character position'),
        bigquery.SchemaField('head_end_char', 'INTEGER', description='Target word end character position')
    ]

    # Column order for Dependency Parsing table
    depparse_column_order = [
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
    ]

    # Morphology schema
    morphology_schema = [
        bigquery.SchemaField('identifier', 'STRING', description='Identifier for the record'),
        bigquery.SchemaField('sentence_num', 'INTEGER', description='Sentence number'),
        bigquery.SchemaField('word_num', 'INTEGER', description='Word number in the sentence'),
        bigquery.SchemaField('word_id', 'STRING', description='Word identifier'),
        bigquery.SchemaField('word', 'STRING', description='Word text'),
        bigquery.SchemaField('lemma', 'STRING', description='Lemma of the word'),
        bigquery.SchemaField('features_Number', 'STRING', description='Number feature'),
        bigquery.SchemaField('features_Mood', 'STRING', description='Mood feature'),
        bigquery.SchemaField('features_Person', 'STRING', description='Person feature'),
        bigquery.SchemaField('features_Tense', 'STRING', description='Tense feature'),
        bigquery.SchemaField('features_VerbForm', 'STRING', description='Verb form feature'),
        bigquery.SchemaField('features_Case', 'STRING', description='Case feature'),
        bigquery.SchemaField('features_Gender', 'STRING', description='Gender feature'),
        bigquery.SchemaField('features_PronType', 'STRING', description='Pronoun type feature'),
        bigquery.SchemaField('features_Degree', 'STRING', description='Degree feature'),
        bigquery.SchemaField('features_Definite', 'STRING', description='Definite feature'),
        bigquery.SchemaField('features_NumForm', 'STRING', description='Number form feature'),
        bigquery.SchemaField('features_NumType', 'STRING', description='Number type feature'),
        bigquery.SchemaField('features_Voice', 'STRING', description='Voice feature'),
        bigquery.SchemaField('start_char', 'INTEGER', description='Start character position in text'),
        bigquery.SchemaField('end_char', 'INTEGER', description='End character position in text')
    ]

    morphology_column_order = [
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
    ]

class PushTables:

    def prepare_chunk_for_push(self, result_dfs, processor_name, library):
        dfs_concat = [inner_list[0] for inner_list in result_dfs]
        results_dfs_concat = pd.concat(dfs_concat, ignore_index=True)

        # Write results_dfs_concat to csv
        logging.info(f'Writing {processor_name} output to csv...')

        results_dfs_concat.to_csv(
            f'TextAnalyticsPipeline/temp/temp_{processor_name}_{library}.csv',
            encoding='utf-8',
            index=False
        )

    def push_to_gbq(self, database_import, bq, project, dataset, table, table_schema, library, logging, proc):

        logging.info(f'Checking if dataset {dataset} exists...')

        dataset = f'{project}.{dataset}'
        schema = table_schema

        # Create dataset if does not exist
        try:
            bq.get_dataset(dataset)  # Make an API request.
            logging.info(f'Dataset {dataset} already exists. Pushing to existing dataset...')
        except NotFound:
            logging.info(f'Dataset {dataset} is not found. Creating new dataset.')
            bq.create_dataset(dataset)
            logging.info(f'Created new dataset: {dataset}.')

        if os.path.isfile(f'TextAnalyticsPipeline/temp/temp_{proc}_{library}.csv') == True:
            logging.info(f'Pushing {proc} to BigQuery dataset: {dataset}...')

            table_id = bigquery.Table(f'{dataset}.{table}')
            try:
                bq.get_table(table_id)
                logging.info('Table exists')
            except:
                table_ = bq.create_table(table_id)
                logging.info(f'Created table {table_.project}.{table_.dataset_id}.{table_.table_id}')

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,
                schema=schema,
                max_bad_records=0
            )

            job_config.allow_quoted_newlines = True

            with open(f'TextAnalyticsPipeline/temp/temp_{proc}_{library}.csv', 'rb') as fh:
                records = sum(1 for line in fh)
                fh.seek(0)
                if database_import == True:
                    if table_schema == Schema.ner_schema:
                        suff = 'named_entities'
                    elif table_schema == Schema.pos_schema:
                        suff = 'part_of_speech'
                    elif table_schema == Schema.depparse_schema:
                        suff = 'depparse'
                    elif table_schema == Schema.morphology_schema:
                        suff = 'morphology'
                    else:
                        suff = 'sentiment'
                else:
                    suff = ''

                job = bq.load_table_from_file(fh, f'{dataset}.{table}_{library}_{suff}', job_config=job_config)
                job.result()  # Waits for the job to complete.


            table = bq.get_table(table_id)
            logging.info(
                f"Loaded {records} rows and {len(table.schema)} columns to {table.project}.{table.dataset_id}.{table.table_id}_{library}_{suff}")
                # Todo (Records)

            os.remove(f'TextAnalyticsPipeline/temp/temp_{proc}_{library}.csv')

            logging.info(f'Results of {proc} successfully pushed to BigQuery!\n')
