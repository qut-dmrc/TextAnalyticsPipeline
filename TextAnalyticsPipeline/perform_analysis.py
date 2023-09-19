'''
Uses Stanza to detect entities in a body of text
Documentation: https://stanfordnlp.github.io/stanza/
'''

# Import Statements
import stanza
import os
import logging
import glob

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery.client import Client
from google.api_core import exceptions

# local imports
from .config import BigQuery, InputConf, ProcessorClass
from .bigquery_tools import PushTables, StanzaSchema #, NLTKSchema, SpaCySchema
from .data_processor import ProcessResults
from .set_up_logging import set_up_logging
from .validate_params import ValidateParams

# stanza.download('en') # download English model

# Google BigQuery Config
# ----------------------------------------------------------------------------------------------------------------------



def run_text_pipeline():

    set_up_logging('TextAnalyticsPipeline/logs')

    # Get Google BigQuery config
    gbq = BigQuery()
    inp = InputConf()

    project = gbq.project_name
    dataset = gbq.dataset_name
    table = gbq.tablename
    id_column = inp.id_column
    text_column = inp.text_column

    # Credentials
    try:
        gbq_creds = os.environ['gbq_servicekey']
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gbq_creds
    except:
        # Check for Google access key; looks for .json service account key in the 'access_key' dir
        servicekeypath = glob.glob(f'{os.getcwd()}/TextAnalyticsPipeline/access_key/*.json')
        for potential_key in servicekeypath:
            with open(potential_key, 'r') as f:
                contents = f.read()
                if f'"project_id": "{project}"' in contents:
                    gbq_creds = potential_key
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gbq_creds


    # Client
    client = bigquery.Client()
    bq = Client(project=project)

    database_import = inp.from_database



    if database_import == True:
        # Prepare SQL query
        query_string = f"""
            SELECT
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

    pc = ProcessorClass()
    processor_class, mwt, lang = pc.get_processor_class()


    # Initialize empty list to store entities dataframes
    result_dfs = []

    count = 0
    for id, document in zip(identifiers, documents):
        count = count + 1
        # Initialize English neural pipeline
        nlp = stanza.Pipeline(f'{lang}', processors=f'tokenize,{mwt}{processor_class}', download_method=None)
        doc = nlp(document)

        print(
            f'Document ID: {id}\n',
            f'Document Text: {document}'
        )

        # Initialize result processor
        result_processor = ProcessResults()

        if processor_class == 'ner':
            entities = doc.entities
            table_schema = StanzaSchema.ner_schema
            # Process document entities
            logging.info('Processing documents for entity extraction...')
            if len(doc.entities) > 0:
                entities_df = result_processor.process_entities(id, entities)
                result_dfs.append([entities_df])
            else:
                logging.info('No entities found in document.\n')


        elif processor_class == 'lemma, pos, depparse':
            sentences = doc.sentences
            table_schema = [StanzaSchema.sentences_schema, StanzaSchema.dependencies_schema]
            # Process document sentences
            logging.info('Processing sentences for part-of-speech extraction...')
            sentences_df, dependencies_df = result_processor.process_sentences(id, sentences, count)
            result_dfs.append([sentences_df, dependencies_df])

        else:
            result_dfs = None


        # If result_dfs > 5000 rows, push to BigQuery, temp value for testing
        if len(result_dfs) > 10:
            # If len(result_dfs[0] == 1, concatenate result_dfs[0], but if len(result_dfs[0]) > 1, concatenate all dfs in result_dfs[0]
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
                push_tables.push_to_gbq(database_import, bq, project, dataset, table, table_schema, proc='entities')
                logging.info('Entities successfully pushed to BigQuery!\n')
                # empty result_dfs list
                result_dfs = []

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

            # finally:
                # # push the remaining dfs to BigQuery
                # # Concatenate sentences dfs
                # result_dfs_sent = []
                # for i in range(len(result_dfs)):
                #     result_dfs_sent.append(result_dfs[i][0])
                # sentences_dfs_concat = pd.concat(result_dfs_sent, axis=0)
                #
                # # Column order
                # # Set sentences_df column order as Schema.sentences_column_order. If column not in df, add it and set value to nan
                # sentences_column_order = Schema.sentences_column_order
                # for column in sentences_column_order:
                #     if column not in sentences_dfs_concat.columns:
                #         sentences_dfs_concat[column] = pd.NA
                # sentences_dfs_concat = sentences_dfs_concat[sentences_column_order]
                #
                # # Write sentences_df to csv
                # logging.info('Writing sentences to csv...')
                # sentences_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_sentences_stanza.csv', encoding='utf-8',
                #                             index=False)
                # records = len(sentences_dfs_concat)
                # # Push sentences to BigQuery
                # push_tables = PushTables()
                # logging.info('Pushing sentences to BigQuery...')
                # push_tables.push_to_gbq(database_import, bq, project, dataset, table, records,
                #                         table_schema=StanzaSchema.sentences_schema, proc='sentences')
                # logging.info('Sentences successfully pushed to BigQuery!\n')
                #
                # # Concatenate dependencies dfs
                # result_dfs_depparse = []
                # for i in range(len(result_dfs)):
                #     result_dfs_depparse.append(result_dfs[i][1])
                # dependencies_dfs_concat = pd.concat(result_dfs_depparse, axis=0)
                #
                # # Set dependencies_df column order as Schema.dependencies_column_order. If column not in df, add it and set value to nan
                # dependencies_column_order = Schema.dependencies_column_order
                # for column in dependencies_column_order:
                #     if column not in dependencies_dfs_concat.columns:
                #         dependencies_dfs_concat[column] = pd.NA
                #
                # dependencies_dfs_concat = dependencies_dfs_concat[dependencies_column_order]
                # # Write dependencies_df to csv
                # logging.info('Writing dependencies to csv...')
                # dependencies_dfs_concat.to_csv('TextAnalyticsPipeline/temp/temp_depparse_stanza.csv', encoding='utf-8',
                #                                index=False)
                # records = len(dependencies_dfs_concat)
                # # Push dependencies to BigQuery
                # push_tables = PushTables()
                # logging.info('Pushing dependencies to BigQuery...')
                # push_tables.push_to_gbq(database_import, bq, project, dataset, table, records,
                #                         table_schema=StanzaSchema.dependencies_schema, proc='depparse')
                # logging.info('Dependencies successfully pushed to BigQuery!\n')
                # # empty result_dfs list
                # result_dfs = []

    exit()
