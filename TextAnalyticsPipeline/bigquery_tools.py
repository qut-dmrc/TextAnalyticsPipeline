'''
This file contains the Google BigQuery table schema for the entity extraction output table, and the function to push the
output to BigQuery.
'''

import os

from google.cloud import bigquery
from google.cloud.bigquery.client import Client
from google.cloud.exceptions import NotFound

from .config import BigQuery
from .set_up_logging import *


class StanzaSchema:

    # Named Entity Recognition schema
    ner_schema = [
        bigquery.SchemaField("identifier", "STRING", mode="REQUIRED", description="URL to the news article (LINK to dataset of origin)."),
        bigquery.SchemaField("text", "STRING", mode="NULLABLE", description="Named entity text"),
        bigquery.SchemaField("type", "STRING", mode="NULLABLE", description="Named entity type"),
        bigquery.SchemaField("start_char", "INTEGER", mode="NULLABLE", description="Location of start character of entity in input string."),
        bigquery.SchemaField("end_char", "INTEGER", mode="NULLABLE", description="Location of end character of entity in input string.")
    ]

    # Column order for named entities table
    ner_column_order = [
        'identifier',
        'text',
        'type',
        'start_char',
        'end_char'
    ]

    # Part of Speech Tagging schema
    sentences_schema = [
        bigquery.SchemaField('identifier', 'STRING', description='Identifier for the record'),
        bigquery.SchemaField('sentence_num', 'INTEGER', description='Sentence number'),
        bigquery.SchemaField('word_num', 'INTEGER', description='Word number in the sentence'),
        bigquery.SchemaField('word_id', 'STRING', description='Word identifier'),
        bigquery.SchemaField('word', 'STRING', description='Word text'),
        bigquery.SchemaField('lemma', 'STRING', description='Lemma of the word'),
        bigquery.SchemaField('upos', 'STRING', description='Universal Part-of-Speech tag'),
        bigquery.SchemaField('xpos', 'STRING', description='Language-specific Part-of-Speech tag'),
        bigquery.SchemaField('head', 'STRING', description='Head word identifier'),
        bigquery.SchemaField('deprel', 'STRING', description='Dependency relation to the head word'),
        bigquery.SchemaField('start_char', 'INTEGER', description='Start character position in text'),
        bigquery.SchemaField('end_char', 'INTEGER', description='End character position in text'),
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
        bigquery.SchemaField('features_Voice', 'STRING', description='Voice feature')
    ]

    # Column order for sentences table
    sentences_column_order = [
        'identifier',
        'sentence_num',
        'word_num',
        'word_id',
        'word',
        'lemma',
        'upos',
        'xpos',
        'head',
        'deprel',
        'start_char',
        'end_char',
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
        'features_Voice'
        ]

    # Dependency Parsing schema
    dependencies_schema = [
        bigquery.SchemaField('identifier', 'STRING', description='Identifier for the record'),
        bigquery.SchemaField('sentence_num', 'INTEGER', description='Sentence number'),
        bigquery.SchemaField('word_num', 'INTEGER', description='Source word identifier'),
        bigquery.SchemaField('word_id', 'STRING', description='Source word identifier'),
        bigquery.SchemaField('word_text', 'STRING', description='Source word text'),
        bigquery.SchemaField('word_lemma', 'STRING', description='Source word lemma'),
        bigquery.SchemaField('relation', 'STRING', description='Dependency relation'),
        bigquery.SchemaField('head_id', 'STRING', description='Target word identifier'),
        bigquery.SchemaField('head_text', 'STRING', description='Target word text'),
        bigquery.SchemaField('head_lemma', 'STRING', description='Target word lemma'),
        bigquery.SchemaField('head_upos', 'STRING', description='Target word Universal Part-of-Speech tag'),
        bigquery.SchemaField('head_xpos', 'STRING', description='Target word language-specific Part-of-Speech tag'),
        bigquery.SchemaField('head_head', 'INTEGER', description='Target word head identifier'),
        bigquery.SchemaField('head_deprel', 'STRING', description='Target word dependency relation'),
        bigquery.SchemaField('head_start_char', 'INTEGER', description='Target word start character position'),
        bigquery.SchemaField('head_end_char', 'INTEGER', description='Target word end character position'),
        bigquery.SchemaField('head_feats_Number', 'STRING', description='Target word Number feature'),
        bigquery.SchemaField('head_feats_Mood', 'STRING', description='Target word Mood feature'),
        bigquery.SchemaField('head_feats_Person', 'STRING', description='Target word Person feature'),
        bigquery.SchemaField('head_feats_Tense', 'STRING', description='Target word Tense feature'),
        bigquery.SchemaField('head_feats_VerbForm', 'STRING', description='Target word Verb form feature'),
        bigquery.SchemaField('head_feats_Case', 'STRING', description='Target word Case feature'),
        bigquery.SchemaField('head_feats_Gender', 'STRING', description='Target word Gender feature'),
        bigquery.SchemaField('head_feats_PronType', 'STRING', description='Target word Pronoun type feature'),
        bigquery.SchemaField('head_feats_Degree', 'STRING', description='Target word Degree feature'),
        bigquery.SchemaField('head_feats_Definite', 'STRING', description='Target word Definite feature'),
        bigquery.SchemaField('head_feats_NumForm', 'STRING', description='Target word Number form feature'),
        bigquery.SchemaField('head_feats_NumType', 'STRING', description='Target word Number type feature'),
        bigquery.SchemaField('head_feats_Voice', 'STRING', description='Target word Voice feature')
    ]

    # Column order for dependencies table
    dependencies_column_order = [
        'identifier',
        'sentence_num',
        'word_num',
        'word_id',
        'word_text',
        'word_lemma',
        'relation',
        'head_id',
        'head_text',
        'head_lemma',
        'head_upos',
        'head_xpos',
        'head_head',
        'head_deprel',
        'head_start_char',
        'head_end_char',
        'head_feats_Number',
        'head_feats_Mood',
        'head_feats_Person',
        'head_feats_Tense',
        'head_feats_VerbForm',
        'head_feats_Case',
        'head_feats_Gender',
        'head_feats_PronType',
        'head_feats_Degree',
        'head_feats_Definite',
        'head_feats_NumForm',
        'head_feats_NumType',
        'head_feats_Voice'
    ]




class PushTables:

    def push_to_gbq(self, database_import, bq, project, dataset, table, records, table_schema, proc):
        dataset = f"{project}.{dataset}"
        schema = table_schema

        # Create dataset if does not exist
        try:
            bq.get_dataset(dataset)  # Make an API request.
            logging.info(f"Dataset {dataset} already exists")
        except NotFound:
            logging.info(f"Dataset {dataset} is not found")
            bq.create_dataset(dataset)
            logging.info(f"Created new dataset: {dataset}")

        if os.path.isfile(f'TextAnalyticsPipeline/temp/temp_{proc}_stanza.csv') == True:
            logging.info('Pushing temp file to BigQuery dataset...')

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

            with open(f'TextAnalyticsPipeline/temp/temp_{proc}_stanza.csv', 'rb') as fh:
                if database_import == True:
                    if table_schema == StanzaSchema.ner_schema:
                        suff = '_named_entities'
                    elif table_schema == StanzaSchema.sentences_schema:
                        suff = '_part_of_speech_'
                    elif table_schema == StanzaSchema.dependencies_schema:
                        suff = '_depparse_'
                    else:
                        suff = '_sentiment'
                else:
                    suff = ''

                job = bq.load_table_from_file(fh, f'{dataset}.{table}{suff}', job_config=job_config)
                job.result()  # Waits for the job to complete.

            table = bq.get_table(table_id)
            logging.info(
                f"Loaded {records} rows and {len(table.schema)} columns to {table.project}.{table.dataset_id}.{table.table_id}{suff}")

            os.remove(f'TextAnalyticsPipeline/temp/temp_{proc}_stanza.csv')
