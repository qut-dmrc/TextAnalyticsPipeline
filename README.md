# Text Analytics Pipeline

The Text Analytics Pipeline is a versatile toolkit that integrates several natural language processing libraries, including spaCy, NLTK, Stanza, and CoreNLP. Its primary purpose is to streamline the extraction, processing and storage of information from text data.

Key functionalities of the Text Analytics Pipeline include:

1. **Named Entity Recognition (NER)**: Automatically identifies and categorizes entities such as names of people, places and organizations within the text.
2. **Part of Speech Tagging (POS)**: Assigns grammatical parts of speech to each word in the text.
3. **Dependency Parsing**: Analyzes the grammatical structure of sentences, establishing relationships between words and their dependencies.
4. **TBC - sentiment analysis?**

The pipeline executes these processes andReadme transfers the cleaned and structured data to a designated Google BigQuery database.
###
### Requirements
- Python 3.10
- Google BigQuery Service Account Key
- A pre-processed Google BigQuery table with an ID column and a document text column. Example:

   | document_id | document_text                                                                                                          |
   |------------------------------------------------------------------------------------------------------------------------| -------------- |
   | 1234567890 | This is a document.                                                                                                    |
   | 1234567891 | This is another document. Documents can be comprised of multiple sentences, depending on the purpose of your analysis. |
   | 1234567892 | But each document should not exceed 1,000,000 (a million) characters.|                                                                                              

- Alternatively, you can provide a csv file to be analysed, as long as you specify `from_database: True` in your `config.yml` file.
  - csv files must be placed in the `/input_csv` directory.
###
### Installation
1. Clone the repository `git clone https://github.com/qut-dmrc/TextAnalyticsPipeline.git`
2. Configure your virtual environment and install the required packages using the following command:
   pip `install -r requirements.txt`
3. `git checkout` collab_branch to move to the collaboration branch. Always push to this branch.
###
### Usage
   1. Ensure your Google BigQuery table is pre-processed and ready for analysis.
   2. Ensure your Google BigQuery Service Account Key is saved in the `/access_key` folder.
   3. Create a config.yml file in the `/config` directory. Use `config_template.yml` as a template. Your specific use case will determine which library you use. For example, if you want to use the pipeline to extract Named Entities from text, you can set your named_entities to `True`. An example config is provided below:
      ```
      # Google BigQuery Params
      
      servicekey_path: ''                             # Path to Google BigQuery Credentials
      project_name: 'my-project'                      # Name of your Google BigQuery project
      dataset_name: 'cool_dataset'                    # Name of dataset containing the table to be analysed, also the destination dataset for the output table
      tablename: 'my_preprocessed_documents'          # Name of table containing the documents to be analysed, the process name will be appended to this name to generate a new table for e.g. _named_entities
      
      id_column: 'comment_id'                         # Name of the column containing the document IDs
      text_column: 'comment_text'                     # Name of the column containing the document text
      
      from_database: True                             # Set to True if you want to analyse a table in Google BigQuery, otherwise set to False
      from_csv: False                                 # Set to True if you want to analyse a csv file, otherwise set to False
      
      
      # Pipeline Params
      
      language: 'en'                                  # Language of the documents to be analysed (see below for supported languages)
      
      named_entity_recognition: False                 # Set to True if you want to extract named entities from the text, otherwise set to False
      part_of_speech: True                            # Set to True if you want to extract part of speech tags from the text and run dependency parsing, otherwise set to False
      sentiment: False                                # Set to True if you want to extract sentiment from the text, otherwise set to False
      
      stanza: True                                    # Set to True if you want to use stanza, otherwise set to False
      spacy: False                                    # Set to True if you want to use spaCy, otherwise set to False
      nltk: False                                     # Set to True if you want to use NLTK, otherwise set to False
      ```
4. Run `run_pipeline.py` to run the pipeline. 
5. I recommend running on a virtual machine if possible. The pipeline can take a while to run, depending on the size of your dataset and the number of processes you are running. 
###
### Output
Todo
###
### Tool recommendations by language
Todo
###
### Collaborators
Ensure you git checkout collab_branch before you start working on the project.
When you are done, push your changes to the collab_branch.