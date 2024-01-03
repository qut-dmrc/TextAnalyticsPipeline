'''
Uses Stanza, Spacy and NLTK to process different text analysis tasks

Documentation: 
https://stanfordnlp.github.io/stanza/
https://spacy.io/
https://www.nltk.org/

'''

# Import Statements


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
from .bigquery_tools import GBQCreds, QueryGBQ
from .set_up_logging import set_up_logging
from .validate_params import ValidateParams

from .stanza import run_stanza_pipeline
from .spacy import run_spacy_pipeline


def get_processor_params():
    # Get the processor class and processor name to be used (see config.py)
    pc = ProcessorClass()
    processor_class, processor_name = pc.get_processor_class()

    # Get the language to be used (see config.py)
    lg = Language()
    lang = lg.get_language()

    # Get the library to be used (see config.py)
    lb = Library()
    library = lb.get_library()

    return processor_class, processor_name, lang, library

def get_input_params():
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

    return project, dataset, table, id_column, text_column, database_import

def find_optimal_chunk_size(a, min_chunk_size=5000, max_chunk_size=10000):
    '''
    Finds the optimal chunk size for a dataframe of length a, given a minimum and maximum chunk size.
    Also prevents the last chunk from being ignored and not processed if it is smaller than the chunk size.
    '''
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


# Main -----------------------------------------------------------------------------------------------------------------

def run_text_pipeline():
    '''
    Runs the text analysis pipeline; the starting point for the pipeline.
    '''

    # Get processor and input parameters from config.yml
    processor_class, processor_name, lang, library = get_processor_params()
    project, dataset, table, id_column, text_column, database_import = get_input_params()

    # Set up logging (see set_up_logging.py)
    set_up_logging('TextAnalyticsPipeline/logs', library, processor_name)

    # Get Google BigQuery credentials
    gbq_creds = GBQCreds()
    gbq_creds.get_gbq_creds(project)

    # Initialise BigQuery client
    bq = gbq_creds.get_client(project)

    # Validate GBQ parameters
    vdp = ValidateParams()
    project, dataset, table = vdp.validate_project_parameters(project, dataset, table, bq)


    # If database_import is True, query BigQuery table. If False, read csv into dataframe
    gbqq = QueryGBQ()
    if database_import == True:
        # Prepare SQL query
        query_string = f"""
            SELECT DISTINCT
            {id_column}, {text_column}
            FROM `{project}.{dataset}.{table}`
            """
        n_docs, df = gbqq.query_gbq(logging, table, query_string, bq, dataset, text_column)
    else:
        n_docs, df = gbqq.read_csv_from_file(logging)


    # Specify a chunk size so that when result_dfs reaches chunk size, it is pushed to BigQuery
    chunk = find_optimal_chunk_size(
        n_docs
    )
    # chunk = 20

    # Get identifiers and documents from dataframe
    identifiers = df[id_column].tolist()
    documents = df[text_column].tolist()

    # Initialise result_dfs (starter is blank list)
    result_dfs = []

    if library == 'stanza':
        run_stanza_pipeline(
            chunk,
            n_docs,
            bq,
            identifiers,
            documents,
            lang,
            library,
            processor_class,
            processor_name,
            logging,
            database_import,
            project,
            dataset,
            table,
            result_dfs
        )

    elif library == 'spacy':
        run_spacy_pipeline(
            chunk,
            n_docs,
            bq,
            identifiers,
            documents,
            lang,
            library,
            processor_class,
            processor_name,
            logging,
            database_import,
            project,
            dataset,
            table,
            result_dfs
        )

    elif library == 'nltk':
        run_nltk_pipeline(
            chunk,
            n_docs,
            bq,
            identifiers,
            documents,
            lang,
            library,
            processor_class,
            processor_name,
            logging,
            database_import,
            project,
            dataset,
            table,
            result_dfs
        )

    elif library == 'corenlp':
        run_corenlp_pipeline(
            chunk,
            n_docs,
            bq,
            identifiers,
            documents,
            lang,
            library,
            processor_class,
            processor_name,
            logging,
            database_import,
            project,
            dataset,
            table,
            result_dfs
        )

    logging.info(f'{library} {processor_name} processing complete!')
    exit()
