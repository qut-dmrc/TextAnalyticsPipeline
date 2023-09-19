# Text Analytics Pipeline

The Text Analytics Pipeline is a versatile toolkit that integrates several natural language processing libraries, including spaCy, NLTK, Stanza, and CoreNLP. Its primary purpose is to empower users to streamline the extraction, processing and storage of information from textual data.

Key functionalities of the Text Analytics Pipeline include:

1. **Named Entity Recognition (NER)**: Automatically identifies and categorizes entities such as names of people, places and organizations within the text.
2. **Part of Speech Tagging (POS)**: Assigns grammatical parts of speech to each word in the text.
3. **Dependency Parsing**: Analyzes the grammatical structure of sentences, establishing relationships between words and their dependencies.
4. **TBC - sentiment analysis?**
5. 

The pipeline executes these processes and  transfers the cleaned and structured data to a designated Google BigQuery database.

### Requirements
- Python 3.10
- Google BigQuery Service Account Key
- 

### Installation
1. Clone the repository `git clone [repo url]`
2. Configure your virtual environment and install the required packages using the following command:
   pip `install -r requirements.txt`
3. `git checkout` collab_branch

### Usage
Your specific use case will determine which library you use. For example, if you want to use the pipeline to extract Named Entities from text, you can set your configuration in `config.yml`.

Run `run pipeline.py` to run the pipeline. 

### Output
Todo

### Collaborators
Always git pull from the master branch. Ensure you git checkout collab_branch before you start working on the project.
When you are done, push your changes to the collab_branch.