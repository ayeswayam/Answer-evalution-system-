# ploomber_task.py
"""
Ploomber AI task definition file
This file defines the pipeline tasks for Ploomber AI
"""
from pathlib import Path
import pandas as pd
from answer_evaluation import evaluate_answers
def evaluate_task(product):
"""Task to evaluate answers from CSV file"""
# The input file path should be provided by Ploomber
input_file = product['upstream']['data']
# Run evaluation
results = evaluate_answers(input_file)
# Save the results
results.to_csv(product['nb'], index=False)
return product
