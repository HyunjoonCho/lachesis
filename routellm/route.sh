# python experiment.py

# by default strong model is set to gpt-4o
# python experiment.py -m2 llama3.1-8b --model2_result d4j_eol_llama3.1_R10.json -r 10
# python experiment.py -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R10.json -r 10
# python experiment.py -m2 qwen2.5-coder-7b --model2_result d4j_eol_qwen2.5-coder_R10.json -r 10

# by default weak model is set to llama3, but it turned out that it won llama3.1 and qwen2.5-coder for R=10
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R3.json -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R3.json -r 3
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R5.json -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R5.json -r 5
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R7.json -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R7.json -r 7
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R15.json -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R15.json -r 15
python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R20.json -m2 mistral-nemo-12b --model2_result d4j_eol_mistral-nemo_R20.json -r 20

#python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R10.json -m2 llama3.1-8b --model2_result d4j_eol_llama3.1_R10.json -r 10
#python experiment.py -m1 llama3-8b --model1_result d4j_eol_llama3_R10.json -m2 qwen2.5-coder-7b --model2_result d4j_eol_qwen2.5-coder_R10.json -r 10
