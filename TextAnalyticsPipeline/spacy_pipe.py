import spacy
import pandas as pd

from .bigquery_tools import Schema, PushTables
from .data_processor import ProcessResults

def run_spacy_pipeline(chunk, n_docs, bq, identifiers, documents, lang, library, processor_class, processor_name, logging, database_import, project, dataset, table, result_dfs):
    # Initialize the Spacy model
    nlp = spacy.load(f'{lang}_core_web_lg')

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
            logging.info(f'Processing document id: {id}')

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

                # Check len of result_dfs and if len(result_dfs) == chunk, push chunk to BigQuery
                if len(result_dfs) == chunk or count > n_docs - chunk:
                    push_tables = PushTables()

                    push_tables.prepare_chunk_for_push(result_dfs, processor_name, library)

                    # Push results_dfs_concat to BigQuery
                    push_tables.push_to_gbq(
                        database_import,
                        bq,
                        project,
                        dataset,
                        table,
                        table_schema,
                        library,
                        logging,
                        proc=processor_name
                    )

                    # Reset result_dfs
                    result_dfs = []

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
            logging.info(f'Processing document id: {id}')

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

            # Check len of result_dfs and if len(result_dfs) == chunk, push chunk to BigQuery
            if len(result_dfs) == chunk or count > n_docs - chunk:
                push_tables = PushTables()

                push_tables.prepare_chunk_for_push(result_dfs, processor_name, library)

                # Push results_dfs_concat to BigQuery
                push_tables.push_to_gbq(
                    database_import,
                    bq,
                    project,
                    dataset,
                    table,
                    table_schema,
                    library,
                    logging,
                    proc=processor_name
                )

                # Reset result_dfs
                result_dfs = []

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
            logging.info(f'Processing document id: {id}')

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

            # Check len of result_dfs and if len(result_dfs) == chunk, push chunk to BigQuery
            if len(result_dfs) == chunk or count > n_docs - chunk:
                push_tables = PushTables()

                push_tables.prepare_chunk_for_push(result_dfs, processor_name, library)

                # Push results_dfs_concat to BigQuery
                push_tables.push_to_gbq(
                    database_import,
                    bq,
                    project,
                    dataset,
                    table,
                    table_schema,
                    library,
                    logging,
                    proc=processor_name
                )

                # Reset result_dfs
                result_dfs = []

    elif processor_name == 'morphology':

        # Set table schema
        table_schema = Schema.morphology_schema

        count = 0
        for id, document in zip(identifiers, documents):

            # Count keeps track of the number of documents processed
            count = count + 1

            # Process the document with the Spacy model
            doc = nlp(document)

            # Process depparse
            logging.info('Processing documents for morphology extraction...')
            logging.info(f'Processing document id: {id}')

            # Initialize result processor
            result_processor = ProcessResults()

            # Initialize sentence number
            sentence_num = 1

            # Run extraction
            morphology_info = []
            for sentence in doc.sents:
                word_num = 1  # Initialize word_num for each sentence
                for token in sentence:
                    # Extracting token attributes
                    morphology_row = {
                        'sentence_num': sentence_num,
                        'word_num': word_num,
                        'word_id': f'{id}_{sentence_num}_{word_num}',
                        'word': token.text,
                        'lemma': token.lemma_,
                        'features_Number': token.morph.get('Number'),
                        'features_Mood': token.morph.get('Mood'),
                        'features_Person': token.morph.get('Person'),
                        'features_Tense': token.morph.get('Tense'),
                        'features_VerbForm': token.morph.get('VerbForm'),
                        'features_Case': token.morph.get('Case'),
                        'features_Gender': token.morph.get('Gender'),
                        'features_PronType': token.morph.get('PronType'),
                        'features_Degree': token.morph.get('Degree'),
                        'features_Definite': token.morph.get('Definite'),
                        'features_NumForm': token.morph.get('NumForm'),
                        'features_NumType': token.morph.get('NumType'),
                        'features_Voice': token.morph.get('Voice'),
                        'start_char': token.idx,
                        'end_char': token.idx + len(token),
                    }
                    word_num += 1  # Increment word_num for each word in the sentence

                    for key, value in morphology_row.items():

                        if isinstance(value, list):
                            if len(value) > 0:
                                morphology_row[key] = value[0]
                            else:
                                morphology_row[key] = ''

                    morphology_info.append(morphology_row)
                sentence_num += 1  # Increment sentence_num for each sentence in the document

            # Transform in df
            df = pd.DataFrame(morphology_info)

            # Run processor
            morphology_df = result_processor.process_morphology(id, df)

            # Append results
            result_dfs.append([morphology_df])

            # Check len of result_dfs and if len(result_dfs) == chunk, push chunk to BigQuery
            if len(result_dfs) == chunk or count > n_docs - chunk:
                push_tables = PushTables()

                push_tables.prepare_chunk_for_push(result_dfs, processor_name, library)

                # Push results_dfs_concat to BigQuery
                push_tables.push_to_gbq(
                    database_import,
                    bq,
                    project,
                    dataset,
                    table,
                    table_schema,
                    library,
                    logging,
                    proc=processor_name
                )

                # Reset result_dfs
                result_dfs = []

    else:
        result_dfs = None