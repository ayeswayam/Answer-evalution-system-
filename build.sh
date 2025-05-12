#!/bin/bash
# Ensure pip is up to date
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Optional: Download any NLTK data if needed
python -c "import nltk; nltk.download('punkt')"
