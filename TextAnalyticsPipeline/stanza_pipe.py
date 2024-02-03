import stanza
import pandas as pd
import os

from .bigquery_tools import Schema, PushTables
from .data_processor import ProcessResults

def run_stanza_pipeline(chunk, n_docs, identifiers, documents, lang, library, processor_class, processor_name, logging, result_dfs):
    # Initialize the Stanza model
    nlp = stanza.Pipeline(f'{lang}', processors=f'tokenize,mwt,{processor_class}', download_method=None)

    cdd = os.getcwd()
    csv_file_path = f'{cdd}/TextAnalyticsPipeline/output_csv/{processor_name}_{library}.csv'

    if processor_name == 'ner':

        # Set table schema
        table_schema = Schema.ner_schema

        logging.info('Processing documents for entity extraction...')

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
            logging.info(f'Processing document id: {id}')

            # Test if there are entities in the output, and, if so, process them
            if len(entities) > 0:

                # Convert entities to dataframe and extract dictionary values
                df = pd.DataFrame(entities)

                # Transform the stanza output to a dict
                df[0] = df[0].apply(lambda x: x.to_dict())

                # Extract values from dictionary
                df = pd.json_normalize(df[0])

                # Initialise result processor
                result_processor = ProcessResults()

                # Run processor
                entities_df = result_processor.process_ner(id, df)

                # Check if the CSV file already exists
                if os.path.isfile(csv_file_path):
                    print('Appending results to existing CSV file')

                    # Append to existing CSV
                    entities_df.to_csv(csv_file_path, mode='a', header=False, index=False)
                else:
                    print('Writing results to new CSV file')
                    # Write a new CSV file
                    entities_df.to_csv(csv_file_path, index=False)

            else:
                logging.info('No entities found in document.\n')

    elif processor_name == 'pos':

        # Set table schema
        table_schema = Schema.pos_schema

        logging.info('Processing documents for part-of-speech extraction...')

        count = 0
        for id, document in zip(identifiers, documents):

            # Count keeps track of the number of documents processed
            count = count + 1

            # Process the document with the Stanza model
            doc = nlp(document)

            # Extract the sentences
            sentences = doc.sentences

            # Process document part-of-speech tagging
            logging.info(f'Processing document id: {id}')

            # Initialize result processor
            result_processor = ProcessResults()

            data = []
            for sent_id, sentence in enumerate(doc.sentences, start=1):
                for word_id, word in enumerate(sentence.words, start=1):
                    data.append(
                        [sent_id, word.id, f'{id}_{sent_id}_{word.id}', word.text, word.lemma, word.upos, word.xpos,
                         word.start_char, word.end_char])

            columns = ['sentence_num', 'word_num', 'word_id', 'word', 'lemma', 'upos', 'xpos', 'start_char', 'end_char']
            df = pd.DataFrame(data, columns=columns)

            # Run processor
            pos_df = result_processor.process_pos(id, df)

            # Check if the CSV file already exists
            if os.path.isfile(csv_file_path):
                print('Appending results to existing CSV file')

                # Append to existing CSV
                pos_df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                print('Writing results to new CSV file')
                # Write a new CSV file
                pos_df.to_csv(csv_file_path, index=False)

    elif processor_name == 'depparse':

        # Set table schema
        table_schema = Schema.depparse_schema

        logging.info('Processing documents for dependency parsing...')

        count = 0
        for id, document in zip(identifiers, documents):

            # Count keeps track of the number of documents processed
            count = count + 1

            # Process the document with the Stanza model
            doc = nlp(document)

            # Process depparse
            logging.info(f'Processing document id: {id}')

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

            # Check if the CSV file already exists
            if os.path.isfile(csv_file_path):
                print('Appending results to existing CSV file')

                # Append to existing CSV
                depparse_df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                print('Writing results to new CSV file')
                # Write a new CSV file
                depparse_df.to_csv(csv_file_path, index=False)

    elif processor_name == 'morphology':

        # Set table schema
        table_schema = Schema.morphology_schema

        logging.info('Processing documents for morphology extraction...')

        count = 0
        for id, document in zip(identifiers, documents):

            # Count keeps track of the number of documents processed
            count = count + 1

            # Process the document with the Stanza model
            doc = nlp(document)

            # Process morphology
            logging.info(f'Processing document id: {id}')

            # Initialize result processor
            result_processor = ProcessResults()

            morphology_info = []

            for sent_num, sentence in enumerate(doc.sentences):
                for word_num, word in enumerate(sentence.words, start=1):

                    # Splitting concatenated features into a dictionary
                    feats_dict = {}
                    if word.feats is not None:
                        feats_list = word.feats.split('|')  # Splitting by '|'
                        for feat in feats_list:
                            key, value = feat.split('=')
                            feats_dict[key] = value

                    # Extracting word attributes
                    morphology_row = {
                        'sentence_num': sentence.index + 1,
                        'word_num': word.id,
                        'word_id': f'{id}_{sentence.index + 1}_{word.id}',
                        'word': word.text,
                        'lemma': word.lemma,
                        'features_Number': feats_dict.get('Number') if feats_dict else None,
                        'features_Mood': feats_dict.get('Mood') if feats_dict else None,
                        'features_Person': feats_dict.get('Person') if feats_dict else None,
                        'features_Tense': feats_dict.get('Tense') if feats_dict else None,
                        'features_VerbForm': feats_dict.get('VerbForm') if feats_dict else None,
                        'features_Case': feats_dict.get('Case') if feats_dict else None,
                        'features_Gender': feats_dict.get('Gender') if feats_dict else None,
                        'features_PronType': feats_dict.get('PronType') if feats_dict else None,
                        'features_Degree': feats_dict.get('Degree') if feats_dict else None,
                        'features_Definite': feats_dict.get('Definite') if feats_dict else None,
                        'features_NumForm': feats_dict.get('NumForm') if feats_dict else None,
                        'features_NumType': feats_dict.get('NumType') if feats_dict else None,
                        'features_Voice': feats_dict.get('Voice') if feats_dict else None,
                        'start_char': word.start_char,
                        'end_char': word.end_char
                    }
                    morphology_info.append(morphology_row)

            # Transform in df
            df = pd.DataFrame(morphology_info)

            # Run processor
            morphology_df = result_processor.process_morphology(id, df)

            # Check if the CSV file already exists
            if os.path.isfile(csv_file_path):
                print('Appending results to existing CSV file')

                # Append to existing CSV
                morphology_df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                print('Writing results to new CSV file')
                # Write a new CSV file
                morphology_df.to_csv(csv_file_path, index=False)

    else:
        result_dfs = None
