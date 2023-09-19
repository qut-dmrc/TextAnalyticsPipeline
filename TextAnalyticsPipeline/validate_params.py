'''
Contains functions for validating user input
'''

import pandas as pd
import re

from google.cloud.exceptions import NotFound
from google.cloud.bigquery.client import Client
from google.auth.exceptions import DefaultCredentialsError


pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class ValidateParams:
    def validate_google_access_key(self, cwd, project):
        '''
        Checks that the Google access key supplied is valid for the project specified in config.yml
        '''

        # Check for Google access key; looks for .json service account key in the 'access_key' dir
        for potential_key in glob.glob(f'{cwd}/access_key/*.json'):
            with open(potential_key, 'r') as f:
                contents = f.read()
                if f'"project_id": "{project}"' in contents:
                    return potential_key
                else:
                    # A key for a different project may still work. Try it out.
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = potential_key
                    try:
                        bq = Client(project=project)

                        # see if the project we are looking for is in the list of available projects
                        available_projects = bq.list_projects()
                        for auth_project in available_projects:
                            if str.lower(project) == str.lower(auth_project.friendly_name):
                                return potential_key
                    except (TypeError, DefaultCredentialsError):
                        pass

                    print(
                        f'The Google service key provided in {potential_key} is not valid for the project specified in config.yml.'
                        f'\nPlease ensure it is the correct one for the {project} project.')
                    continue  # move on to next key in the folder

        print(f'Please enter a valid Google service account access key.')
        exit()

    def validate_project_parameters(self, project, dataset, table, bq):
        '''
        Validates project parameters.
        If parameter is invalid, then parameter=None and program will notify user and exit.
        '''


        # Check project name entered; checks len for presence of a project string and checks invalid character. If
        # specified project already exists, ask user to confirm if it is ok to append to that project. Otherwise, exit.
        if len(project) > 0:
            if bool(re.match("^[A-Za-z0-9-]*$", project)) == True:
                project = project
            else:
                project = None
                print('Invalid project name. Project names may contain letters, numbers, dashes and underscores. Exiting.')
                exit()
        else:
            project = None
            print('No project in config. Please enter a valid project name. Exiting')
            exit()

        # Check dataset name entered; checks len for presence of a dataset string and checks invalid characters. If
        # specified dataset already exists, ask user to confirm if it is ok to append to that dataset. Otherwise, exit.
        if len(dataset) > 0:
            if bool(re.match("^[A-Za-z0-9_]*$", dataset)) == True:

                # Check if dataset exists
                try:
                    bq.get_dataset(f'{project}.{dataset}')
                except NotFound:
                    print('Dataset not found. Create? y/n')
                    user_input = input('>>>')
                    if user_input.lower() == 'y':
                        # Create dataset
                        bq.create_dataset(f'{project}.{dataset}')
                        print(f'Dataset {dataset} created in {project}.')
                    else:
                        print('Exiting.')
                        exit()

        else:
            dataset = None
            print('No dataset in config. Please enter a valid dataset name. Exiting')
            exit()

        # Check table name entered; checks len for presence of a table string and checks invalid characters. If
        # specified table already exists, ask user to confirm if it is ok to append to that table. Otherwise, exit.
        if len(table) > 0:
            if bool(re.match("^[A-Za-z0-9_]*$", table)) == True:
                # Check if table exits in dataset
                try:
                    # Make an API request.
                    bq.get_table(f'{project}.{dataset}.{table}')
                except NotFound:
                    print('Table not found. Exiting.')
                    exit()
        else:
            table = None
            print('No table in config. Please enter a valid table name. Exiting.')
            exit()

        return project, dataset, table