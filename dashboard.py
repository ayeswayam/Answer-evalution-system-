import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from answer_evaluation import AnswerEvaluationSystem
st.set_page_config(page_title="Answer Evaluation Dashboard", layout="wide")
st.title("Answer Evaluation System")
# Initialize the evaluator
evaluator = AnswerEvaluationSystem()
# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Single Answer Evaluation", "Batch Evaluation"])
with tab1:
st.header("Evaluate a Single Answer")
col1, col2 = st.columns(2)
with col1:
st.subheader("Input")
student_answer = st.text_area("Student Answer", height=200)
model_answer = st.text_area("Model Answer", height=200)
if st.button("Evaluate"):
if student_answer and model_answer:
# Evaluate the answer
with st.spinner("Evaluating answer..."):
result = evaluator.evaluate_answer(student_answer, model_answer)
# Store result in session state
st.session_state.result = result
else:
st.error("Please provide both student and model answers.")
with col2:
st.subheader("Results")
if 'result' in st.session_state:
result = st.session_state.result
# Display scores
st.metric("Overall Score", f"{result['overall_score']}%")
# Create a DataFrame for the component scores
scores_df = pd.DataFrame({
'Component': ['Content', 'Keywords', 'Grammar', 'Structure'],
'Score': [
result['semantic_score'],
result['keyword_score'],
result['grammar_score'],
result['structure_score']
]
})
# Create a bar chart
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Component', y='Score', data=scores_df, ax=ax)
ax.set_ylim(0, 100)
ax.set_title('Evaluation Scores by Component')
ax.set_ylabel('Score (%)')
# Add score labels on top of bars
for i, v in enumerate(scores_df['Score']):
ax.text(i, v + 2, f"{v:.1f}%", ha='center')
st.pyplot(fig)
# Display feedback
st.subheader("Feedback")
for feedback_line in result['feedback'].split('\n'):
st.write(feedback_line)
with tab2:
st.header("Batch Evaluation")
# File upload
uploaded_file = st.file_uploader("Upload a CSV file with student answers", type="csv")
if uploaded_file:
# Read the CSV file
df = pd.read_csv(uploaded_file)
# Check if required columns exist
required_columns = ['student_answer', 'model_answer']
if not all(col in df.columns for col in required_columns):
st.error(f"CSV file must contain columns: {', '.join(required_columns)}")
else:
st.write(f"Found {len(df)} answers to evaluate")
# Show preview of the data
st.subheader("Data Preview")
st.dataframe(df.head(5))
# Evaluate button
if st.button("Run Batch Evaluation"):
progress_bar = st.progress(0)
results = []
# Process each answer
for i, row in df.iterrows():
# Evaluate answer
result = evaluator.evaluate_answer(row['student_answer'], row['model_answer'])
# Add question ID if available
if 'question_id' in row:
result['question_id'] = row['question_id']
results.append(result)
# Update progress bar
progress_bar.progress((i + 1) / len(df))
# Create results dataframe
results_df = pd.DataFrame(results)
# Store in session state
st.session_state.batch_results = results_df
st.success(f"Evaluated {len(results_df)} answers!")
# Display batch results if available
if 'batch_results' in st.session_state:
st.subheader("Evaluation Results")
results_df = st.session_state.batch_results
# Summary statistics
st.write("### Summary Statistics")
summary_stats = pd.DataFrame({
'Metric': ['Mean', 'Median', 'Min', 'Max'],
'Overall Score': [
f"{results_df['overall_score'].mean():.2f}%",
f"{results_df['overall_score'].median():.2f}%",
f"{results_df['overall_score'].min():.2f}%",
f"{results_df['overall_score'].max():.2f}%"
]
})
st.table(summary_stats)
# Distribution plot
st.write("### Score Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(results_df['overall_score'], bins=10, kde=True, ax=ax)
ax.set_xlabel('Overall Score (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Answer Scores')
st.pyplot(fig)
# Detailed results table
st.write("### Individual Results")
# Add columns for display
display_df = results_df.copy()
if 'question_id' in display_df.columns:
display_df = display_df[['question_id', 'overall_score', 'semantic_score',
'keyword_score', 'grammar_score', 'structure_score']]
else:
display_df = display_df[['overall_score', 'semantic_score',
'keyword_score', 'grammar_score', 'structure_score']]
st.dataframe(display_df)
# Export results option
st.download_button(
label="Download Results CSV",
data=results_df.to_csv(index=False).encode('utf-8'),
file_name="evaluation_results.csv",
mime="text/csv"
)
# Show individual feedback for a selected answer
if 'question_id' in results_df.columns:
selected_id = st.selectbox(
"Select a question to view detailed feedback:",
options=results_df['question_id'].unique()
)
# Get the selected answer
selected_result = results_df[results_df['question_id'] == selected_id].iloc[0]
else:
selected_index = st.number_input(
"Select answer index to view detailed feedback:",
min_value=0,
max_value=len(results_df)-1,
value=0
)
# Get the selected answer
selected_result = results_df.iloc[selected_index]
# Display detailed feedback
st.write("### Detailed Feedback")
st.write(f"**Overall Score:** {selected_result['overall_score']}%")
st.write(f"**Content Score:** {selected_result['semantic_score']}%")
st.write(f"**Keyword Score:** {selected_result['keyword_score']}%")
st.write(f"**Grammar Score:** {selected_result['grammar_score']}%")
st.write(f"**Structure Score:** {selected_result['structure_score']}%")
st.write("### Feedback")
for feedback_line in selected_result['feedback'].split('\n'):
st.write(feedback_line)
# Show student and model answers
st.write("### Student Answer")
st.write(selected_result['student_answer'])
st.write("### Model Answer")
st.write(selected_result['model_answer'])
