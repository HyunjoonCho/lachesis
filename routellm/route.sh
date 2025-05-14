# python experiment.py

# by default strong model is set to gpt-4o
# python experiment.py -m2 llama3.1-8b --model2_result d4j_eol_llama3.1_R10.json -r 10
# python experiment.py -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R10.json -r 10
# python experiment.py -m2 qwen2.5-coder-7b --model2_result d4j_eol_qwen2.5-coder_R10.json -r 10

# by default weak model is set to llama3, but it turned out that it won llama3.1 and qwen2.5-coder for R=10
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R10.json -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R10.json -r 10
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R10.json -m2 llama3.1-8b --model2_result d4j_eol_llama3.1_R10.json -r 10
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R10.json -m2 qwen2.5-coder-7b --model2_result d4j_eol_qwen2.5-coder_R10.json -r 10