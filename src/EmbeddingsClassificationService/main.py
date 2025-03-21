import argparse

# Project Imports
from src.EmbeddingsClassificationService.service import ANNClassifierService


'''
    Always run the complete pipeline in the main function inside try block 
    To catch any exceptions in the pipeline and efficieltly logging them
'''
def main():
    try:
        service = ANNClassifierService()
    except:
        print("An error occured in the FineTune Service. Please check logs for more details")
        #TODO: Add Logging
