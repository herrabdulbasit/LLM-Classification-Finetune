import argparse

# Project Imports
from src.FineTuneService.service import FineTuneService


parser = argparse.ArgumentParser(description="Parse arguments for Fine Tune Service")
parser.add_argument("--mode", type=int, help="Service Mode: train/inference", default="train")
parser.add_argument("--full-finetune", type=bool, help="Add LoRA or Not", default=False)

args = parser.parse_args()

'''
    Always run the complete pipeline in the main function inside try block 
    To catch any exceptions in the pipeline and efficieltly logging them
'''
def main():
    try:
        service = FineTuneService(full_finetune=args.full_finetune, mode=args.mode)
    except:
        print("An error occured in the FineTune Service. Please check logs for more details")
        #TODO: Add Logging
