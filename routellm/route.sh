# python experiment.py

# by default strong model is set to gpt-4o
# python experiment.py -w llama3.1-8b --weak_result d4j_eol_llama3.1_R10.json
# python experiment.py -w mistral-nemo-12b --weak_result d4j_eol_mistral-nemo_R10.json
# python experiment.py -w qwen2.5-coder-7b --weak_result d4j_eol_qwen2.5-coder_R10.json

# by default weak model is set to llama3, but it turned out that it won llama3.1 and qwen2.5-coder for R=10
python experiment.py -w llama3.1-8b --weak_result d4j_eol_llama3.1_R10.json -s llama3-8b --strong_result d4j_eol_llama3_R10.json
# python experiment.py -s mistral-nemo-12b --strong_result d4j_eol_mistral-nemo_R10.json
python experiment.py -w qwen2.5-coder-7b --weak_result d4j_eol_qwen2.5-coder_R10.json -s llama3-8b --strong_result d4j_eol_llama3_R10.json