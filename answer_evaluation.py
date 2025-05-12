"""
Answer Evaluation System
This project evaluates student answers against model answers using NLP techniques and
provides
individual feedback for each answer. It can be deployed on Ploomber AI.
Main components:
1. Semantic text matching using embeddings
2. Keyword-based evaluation
3. Grammar and spelling check
4. Structure analysis
5. Individual feedback generation
"""
import os
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
# Initialize components
try:
nltk.data.find('tokenizers/punkt')
except:
nltk.download('punkt')
try:
nltk.data.find('corpora/stopwords')
except:
nltk.download('stopwords')
class AnswerEvaluationSystem:
def __init__(self):
self._model = None
self._language_tool = None
self.stop_words = set(stopwords.words('english'))
@property
def model(self):
if self._model is None:
self._model = SentenceTransformer('all-MiniLM-L6-v2')
return self._model
@property
def language_tool(self):
if self._language_tool is None:
self._language_tool = language_tool_python.LanguageTool('en-US')
return self._language_tool
def preprocess_text(self, text):
"""Preprocess text for analysis"""
# Convert to lowercase
text = text.lower()
# Remove special characters and extra spaces
text = re.sub(r'[^\w\s]', ' ', text)
text = re.sub(r'\s+', ' ', text).strip()
return text
def extract_keywords(self, text):
"""Extract important keywords from text"""
tokens = word_tokenize(text)
# Remove stopwords
keywords = [word for word in tokens if word not in self.stop_words and len(word) > 2]
return keywords
def get_embedding(self, text):
"""Generate embeddings for text"""
return self.model.encode(text)
def semantic_similarity(self, student_answer, model_answer):
"""Calculate semantic similarity between student and model answers"""
student_embedding = self.get_embedding(student_answer)
model_embedding = self.get_embedding(model_answer)
# Calculate cosine similarity
similarity = cosine_similarity([student_embedding], [model_embedding])[0][0]
return similarity
def keyword_match_score(self, student_answer, model_answer):
"""Calculate keyword match score"""
student_keywords = self.extract_keywords(student_answer)
model_keywords = self.extract_keywords(model_answer)
# Count how many important keywords from model answer are present in student
answer
matches = [keyword for keyword in model_keywords if keyword in student_keywords]
if len(model_keywords) == 0:
return 0
match_score = len(matches) / len(model_keywords)
return match_score
def grammar_check(self, text):
spell = SpellChecker()
words = text.split()
misspelled = spell.unknown(words)
error_count = len(misspelled)
if len(words) > 0:
error_rate = error_count / len(words)
grammar_score = max(0, 1 - (error_rate * 10))
else:
grammar_score = 0
return grammar_score, error_count, list(misspelled)
def structure_analysis(self, text):
"""Analyze text structure (paragraphs, sentence length, etc.)"""
paragraphs = text.split('\n\n')
paragraph_count = len([p for p in paragraphs if p.strip()])
sentences = re.split(r'[.!?]+', text)
sentence_count = len([s for s in sentences if s.strip()])
if sentence_count == 0:
avg_sentence_length = 0
else:
words = word_tokenize(text)
avg_sentence_length = len(words) / sentence_count
# Score structure based on paragraphs and average sentence length
structure_score = min(1.0, (paragraph_count / 3) * 0.5 + (min(15, avg_sentence_length)
/ 15) * 0.5)
return structure_score, paragraph_count, avg_sentence_length
def evaluate_answer(self, student_answer, model_answer):
"""Evaluate student answer against model answer"""
# Preprocess both answers
processed_student = self.preprocess_text(student_answer)
processed_model = self.preprocess_text(model_answer)
# Calculate various scores
semantic_score = self.semantic_similarity(student_answer, model_answer)
keyword_score = self.keyword_match_score(processed_student, processed_model)
grammar_score, error_count, grammar_errors = self.grammar_check(student_answer)
structure_score, para_count, avg_sent_len = self.structure_analysis(student_answer)
# Calculate overall score (weighted average)
overall_score = (
semantic_score * 0.4 +
keyword_score * 0.3 +
grammar_score * 0.2 +
structure_score * 0.1
) * 100
# Round to 2 decimal places
overall_score = round(overall_score, 2)
# Generate individual feedback
feedback = self.generate_feedback(
student_answer,
model_answer,
semantic_score,
keyword_score,
grammar_score,
error_count,
grammar_errors,
structure_score,
para_count,
avg_sent_len
)
results = {
"overall_score": overall_score,
"semantic_score": round(semantic_score * 100, 2),
"keyword_score": round(keyword_score * 100, 2),
"grammar_score": round(grammar_score * 100, 2),
"structure_score": round(structure_score * 100, 2),
"feedback": feedback
}
return results
import gc
gc.collect()
def generate_feedback(self, student_answer, model_answer, semantic_score,
keyword_score,
grammar_score, error_count, grammar_errors, structure_score,
para_count, avg_sent_len):
"""Generate personalized feedback for the student answer"""
feedback = []
# Content feedback
if semantic_score >= 0.8:
feedback.append(" Excellent understanding of the content. Your answer aligns
very well with the expected response.")
elif semantic_score >= 0.6:
feedback.append("✓ Good grasp of the subject matter. Your answer covers most key
points.")
elif semantic_score >= 0.4:
feedback.append(" Partial understanding shown. Consider reviewing the topic to
strengthen your knowledge.")
else:
feedback.append(" Limited alignment with the expected answer. Please revisit the
material.")
# Keyword feedback
missing_keywords = self.identify_missing_keywords(student_answer, model_answer)
if keyword_score >= 0.8:
feedback.append(" Excellent use of key terminology and concepts.")
elif keyword_score >= 0.6:
feedback.append("✓ Good use of terminology, but some key terms are missing.")
else:
if missing_keywords:
feedback.append(f" Consider including these key concepts: {',
'.join(missing_keywords[:5])}.")
else:
feedback.append(" Many important terms and concepts are missing from your
answer.")
# Grammar feedback
if error_count == 0:
feedback.append(" Perfect grammar and spelling.")
elif error_count <= 2:
feedback.append("✓ Few minor grammar or spelling issues.")
elif error_count <= 5:
feedback.append(f" Several grammar/spelling errors detected ({error_count}).
Consider proofreading your work.")
else:
feedback.append(f" Significant grammar/spelling issues ({error_count}) that affect
readability.")
# Structure feedback
if structure_score >= 0.8:
feedback.append(" Well-structured response with good paragraph organization.")
elif structure_score >= 0.5:
feedback.append("✓ Adequate structure, but could use more organization or
paragraph breaks.")
else:
if para_count <= 1:
feedback.append(" Consider breaking your answer into multiple paragraphs for
better readability.")
if avg_sent_len > 25:
feedback.append(" Your sentences are quite long. Consider using shorter,
clearer sentences.")
return "\n".join(feedback)
def identify_missing_keywords(self, student_answer, model_answer):
"""Identify important keywords from model answer missing in student answer"""
processed_student = self.preprocess_text(student_answer)
processed_model = self.preprocess_text(model_answer)
student_keywords = set(self.extract_keywords(processed_student))
model_keywords = self.extract_keywords(processed_model)
# Get frequency of keywords in model answer
model_keyword_freq = Counter(model_keywords)
important_keywords = [k for k, v in model_keyword_freq.items() if v > 1]
# Identify missing important keywords
missing_keywords = [keyword for keyword in important_keywords if keyword not in
student_keywords]
return missing_keywords
def visualize_results(self, results):
"""Create visualization of evaluation results"""
categories = ['Content', 'Keywords', 'Grammar', 'Structure']
scores = [
results['semantic_score'],
results['keyword_score'],
results['grammar_score'],
results['structure_score']
]
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
# Create bar chart
ax = sns.barplot(x=categories, y=scores)
# Add overall score as text
plt.text(
0.5, 0.9,
f'Overall Score: {results["overall_score"]}%',
horizontalalignment='center',
transform=plt.gca().transAxes,
fontsize=14,
bbox=dict(facecolor='white', alpha=0.8)
)
plt.ylim(0, 100)
plt.title('Answer Evaluation Results')
plt.ylabel('Score (%)')
for i, score in enumerate(scores):
plt.text(i, score + 2, f"{score}%", ha='center')
plt.tight_layout()
return plt
# Example pipeline function for Ploomber AI
def evaluate_answers(data_path):
"""Pipeline function to evaluate a batch of answers"""
# Load data
df = pd.read_csv(data_path)
# Initialize evaluator
evaluator = AnswerEvaluationSystem()
results = []
for _, row in df.iterrows():
student_answer = row['student_answer']
model_answer = row['model_answer']
question_id = row.get('question_id', 'unknown')
# Evaluate the answer
evaluation = evaluator.evaluate_answer(student_answer, model_answer)
# Add metadata
evaluation['question_id'] = question_id
evaluation['student_answer'] = student_answer
evaluation['model_answer'] = model_answer
results.append(evaluation)
# Create results dataframe
results_df = pd.DataFrame(results)
# Save results
results_df.to_csv('evaluation_results.csv', index=False)
return results_df
# Function for evaluating a single answer (useful for API endpoints)
def evaluate_single_answer(student_answer, model_answer):
"""Evaluate a single student answer against a model answer"""
evaluator = AnswerEvaluationSystem()
result = evaluator.evaluate_answer(student_answer, model_answer)
return result
# For Ploomber AI deployment
if __name__ == "__main__":
import sys
if len(sys.argv) > 1:
data_path = sys.argv[1]
results = evaluate_answers(data_path)
print(f"Evaluation complete. Results saved to evaluation_results.csv")
else:
print("Please provide a path to the data file.")
