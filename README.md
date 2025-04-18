***RAG Pipeline Setup***

1. Clone the repo.
2. Install and setup python (v3.10).
3. Setup virtual environment.
```
python -m venv rag_env
source rag_env/bin/activate
```
4. Install dependencies.
```
pip install -r requirements.txt
```
5. Install and setup [ollama](https://ollama.com/download).
6. Pull the required llm model(gemma3:1b)
```
ollama pull gemma3:1b
```
7. Run the program 
```
python main.py
```

note: add the query in the query variable in the main.py; the rag data is stored in the rag_sample.csv
