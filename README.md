# Ensemble Construction in Tree Hypothesis Space via LLM-Derived Mixture-of-Priors

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

3. Run the main script:

```bash
python main.py
```

You can try different datasets and configurations by overriding hyperparameters. For example:

```bash
python main.py \
       alg=rf_cls \
       dataset.name=compas \
       evaluator.model=gpt-5-mini
```
