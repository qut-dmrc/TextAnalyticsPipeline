from .data_processor import ProcessResults

import os
import nltk
import pandas as pd
import logging

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')



def run_nltk_pipeline(chunk, n_docs, identifiers, documents, lang, library, processor_class, processor_name, logging, result_dfs):
    # Ensure correct language (NLTK mostly supports English)
    if lang != 'en':
        print('Only English language is supported by this NLTK pipeline!')
        return

    cdd = os.getcwd()
    csv_file_path = f'{cdd}/TextAnalyticsPipeline/output_csv/{processor_name}_{library}.csv'

    if processor_name == 'ner':

        logging.info('Processing documents for entity extraction...')

        for id, document in zip(identifiers, documents):
            # Tokenize sentences
            sentences = nltk.sent_tokenize(document)
            entities_data = []

            print(
                f'Document ID: {id}\n',
                f'Document Text: {document}'
            )

            # Process document entities
            logging.info(f'Processing document id: {id}')

            for sent_id, sentence in enumerate(sentences, start=1):
                # Tokenize words
                words = nltk.word_tokenize(sentence)

                # POS tagging for words in the sentence
                pos_tags = nltk.pos_tag(words)

                # Named entity recognition (NER)
                named_entities = nltk.ne_chunk(pos_tags)

                for entity in named_entities:
                    if hasattr(entity, 'label'):  # Check if it's a named entity
                        entity_row = {
                            'entity': ' '.join([word for word, tag in entity.leaves()]),
                            'entity_type': entity.label(),
                            'sentence_num': sent_id,
                            'document_id': id
                        }
                        entities_data.append(entity_row)

            if entities_data:
                # Create DataFrame
                df = pd.DataFrame(entities_data)
                result_processor = ProcessResults()
                entities_df = result_processor.process_ner(id, df)

                # Write to CSV
                if os.path.isfile(csv_file_path):
                    entities_df.to_csv(csv_file_path, mode='a', header=False, index=False)
                else:
                    entities_df.to_csv(csv_file_path, index=False)

            else:
                logging.info(f'No named entities found in document id: {id}')

    elif processor_name == 'pos':
        logging.info('Processing documents for part-of-speech extraction...')

        for id, document in zip(identifiers, documents):
            # Tokenize sentences
            sentences = nltk.sent_tokenize(document)
            pos_data = []

            print(
                f'Document ID: {id}\n',
                f'Document Text: {document}'
            )


            # Process document entities
            logging.info(f'Processing document id: {id}')

            for sent_id, sentence in enumerate(sentences, start=1):
                # Tokenize words
                words = nltk.word_tokenize(sentence)

                # POS tagging
                pos_tags = nltk.pos_tag(words)

                for word_id, (word, tag) in enumerate(pos_tags, start=1):
                    pos_row = {
                        'sentence_num': sent_id,
                        'word_num': word_id,
                        'word': word,
                        'pos_tag': tag,
                        'document_id': id
                    }
                    pos_data.append(pos_row)

            # Create DataFrame
            df = pd.DataFrame(pos_data)
            result_processor = ProcessResults()
            pos_df = result_processor.process_pos(id, df)

            # Write to CSV
            if os.path.isfile(csv_file_path):
                pos_df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                pos_df.to_csv(csv_file_path, index=False)

    else:
        logging.info(f'Processor "{processor_name}" is not supported with NLTK.')
        result_dfs = None


