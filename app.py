import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import json
from typing import Optional, List
# Import our evaluation system
from answer_evaluation import AnswerEvaluationSystem, evaluate_answers, evaluate_single_answer

app = FastAPI(title="Answer Evaluation System API")
evaluator = AnswerEvaluationSystem()

class AnswerRequest(BaseModel):
    student_answer: str
    model_answer: str

class AnswerResponse(BaseModel):
    overall_score: float
    semantic_score: float
    keyword_score: float
    grammar_score: float
    structure_score: float
    feedback: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Answer Evaluation System API"}

@app.post("/evaluate", response_model=AnswerResponse)
def evaluate_answer(request: AnswerRequest):
    """Evaluate a single answer"""
    try:
        result = evaluate_single_answer(request.student_answer, request.model_answer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/evaluate-batch")
async def evaluate_batch(file: UploadFile = File(...)):
    """Evaluate a batch of answers from a CSV file"""
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check if required columns exist
        required_columns = ['student_answer', 'model_answer']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV file must contain columns: {', '.join(required_columns)}"
            )
            
        # Process each answer
        results = []
        for _, row in df.iterrows():
            result = evaluate_single_answer(row['student_answer'], row['model_answer'])
            # Add question_id if available
            if 'question_id' in row:
                result['question_id'] = row['question_id']
            results.append(result)
            
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation error: {str(e)}")

# Add port binding for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
