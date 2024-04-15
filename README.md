# Text Analytics Pipeline - workshop branch

The Text Analytics Pipeline is a versatile toolkit that integrates several natural language processing libraries, including spaCy, NLTK, Stanza, and CoreNLP. Its primary purpose is to streamline the extraction, processing and storage of information from text data.

Key functionalities of the Text Analytics Pipeline (Workshop Branch) include:

1. **Named Entity Recognition (NER)**: Automatically identifies and categorizes entities such as names of people, places and organizations within the text.
2. **Part of Speech Tagging (POS)**: Assigns grammatical parts of speech to each word in the text.
3. **Dependency Parsing**: Analyzes the grammatical structure of sentences, establishing relationships between words and their dependencies.


The pipeline executes these processes andReadme transfers the cleaned and structured data to a designated Google BigQuery database.
###
### Requirements
- Python 3.10 or newer
- A csv file to be analysed, placed in the `/input_csv` directory (specify `from_database: True` in your `config.yml` file) - sample exists in this location.

   | document_id | document_text                                                                                                          |
   |------------------------------------------------------------------------------------------------------------------------| -------------- |
   | 1234567890 | This is a document.                                                                                                    |
   | 1234567891 | This is another document. Documents can be comprised of multiple sentences, depending on the purpose of your analysis. |
   | 1234567892 | But each document should not exceed 1,000,000 (a million) characters.|                                                                                              


###
### Installation
1. Clone the repository `git clone https://github.com/qut-dmrc/TextAnalyticsPipeline.git`
2. Configure your virtual environment and install the required packages using the following command:
   pip `install -r requirements.txt`
3. `git checkout online_demo_branch`. Always push to this branch if working in this branch.


###
### Usage
   1. Ensure your csv file is pre-processed and ready for analysis.
      1. Create a config.yml file in the `/config` directory. Use `config_template_workshop.yml` as a template. Your specific use case will determine which library you use. For example, if you want to use the pipeline to extract Named Entities from text, you can set your named_entities to `True`. An example config is provided below:
         ```
         id_column: 'article_id'                         # Name of the column containing the document IDs
         text_column: 'content'                          # Name of the column containing the document text
      
         language: 'en'                                  # Language of the documents to be analysed (see below for supported languages)
      
         named_entity_recognition: True                  # Set to True if you want to extract named entities from the text, otherwise set to False
         part_of_speech: False                           # Set to True if you want to extract part of speech tags from the text and run dependency parsing, otherwise set to False
         dependency_parsing: False                       # Set to True if you want to run dependency parsing on the text, otherwise set to False
         sentiment: False                                # Set to True if you want to extract sentiment from the text, otherwise set to False
         morphology: False                               # Set to True if you want to extract morphology from the text, otherwise set to False
      
         stanza: True                                    # Set to True if you want to use stanza, otherwise set to False
         spacy: False                                    # Set to True if you want to use spaCy, otherwise set to False
         nltk: False                                     # Set to True if you want to use NLTK, otherwise set to False
         nltk: False                                     # Set to True if you want to use NLTK, otherwise set to False
            
         ```
   2. Run `run_pipeline.py` to run the pipeline. 
   3. I recommend running on a virtual machine if possible. The pipeline can take a while to run, depending on the size of your dataset and the number of processes you are running. 
###
### Output
Todo
###
### Tool recommendations by language
Todo
###
### Collaborators
Todo