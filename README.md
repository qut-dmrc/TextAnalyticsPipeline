# Text Analytics Pipeline

This repository contains the code for the Text Analytics Pipeline. The pipeline is a collection of tools that can be used to extract information from text. The pipeline is built to utilise the following tools:
- spaCy
- NLTK
- Stanza

The pipeline can be used to extract information from text using the above libraries, to run various processes, including:
- Named Entity Recognition
- Part of Speech Tagging
- Dependency Parsing

The pipeline runs the above processes and pushes the cleaned and structured data to a Google BigQuery database.

### Installation
1. Clone the repository `git clone [repo url]`
2. Configure your virtual environment and install the required packages using the following command:
   pip `install -r requirements.txt`
3. `git checkout` collab_branch

### Usage
Your specific use case will determine which library you use. For example, if you want to use the pipeline to extract Named Entities from text, you can set your configuration in `config.yml`.

Run `run pipeline.py` to run the pipeline. 

### Collaborators
Always git pull from the master branch. Ensure you git checkout collab_branch before you start working on the project.
When you are done, push your changes to the collab_branch.