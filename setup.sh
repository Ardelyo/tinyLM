#!/bin/bash

echo "🤏 TinyLM Setup"
echo "==============="

# Create virtual environment
python -m venv tinylm_env
source tinylm_env/bin/activate  # On Windows: tinylm_env\Scripts\activate

# Install dependencies
pip install torch numpy

echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "1. Train a model: python tinylm.py train"
echo "2. Chat with model: python tinylm.py chat"
echo "3. Interactive mode: python tinylm.py"
