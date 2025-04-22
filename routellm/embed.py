import json
from routers.matrix_factorization.request_wrapper import get_embeddings

results_path = '../AutoFL/results/d4j_autofl_eol_1/llama3'
initial_prompts_path = 'route_data/initial_prompts.json'
embeddings_path = 'route_data/embeddings.json'

if __name__ == "__main__":
    embeddings_dict = dict()
    with open(initial_prompts_path) as f:
        data = json.load(f)
        
    for bug_id in data:
        initial_prompt = data[bug_id]
        embedding = get_embeddings(initial_prompt)
        embeddings_dict[bug_id] = embedding
    
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings_dict, f)