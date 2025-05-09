python experiment.py

python experiment.py -w llama3.1-8b --weak_result d4j_eol_llama3.1_R10.json
python experiment.py -w mistral-nemo-12b --weak_result d4j_eol_mistral-nemo_R10.json
python experiment.py -w qwen2.5-coder-7b --weak_result d4j_eol_qwen2.5-coder_R10.json

python experiment.py -s llama3.1-8b --strong_result d4j_eol_llama3.1_R10.json
python experiment.py -s mistral-nemo-12b --strong_result d4j_eol_mistral-nemo_R10.json
python experiment.py -s qwen2.5-coder-7b --strong_result d4j_eol_qwen2.5-coder_R10.json