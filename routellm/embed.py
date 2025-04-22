import json
import requests
import time

results_path = '../AutoFL/results/d4j_autofl_eol_1/llama3'
initial_prompts_path = 'route_data/initial_prompts.json'
endpoint = 'http://localhost:11434/api/embeddings'
embeddings_path = 'route_data/embeddings.json'

def _query_model(payload):
    for _ in range(5):
        try:
            json_payload = json.dumps(payload)
            headers = {'Content-Type': 'application/json'}
            response = json.loads(requests.post(endpoint, data=json_payload, headers=headers).text)
            return response['embedding']
        except Exception as e:
            save_err = e
            if "The server had an error processing your request." in str(e):
                time.sleep(1)
            else:
                break
    raise save_err


def get_LLM_response(initial_prompt):
    payload = {
        'model': 'nomic-embed-text',
        'prompt': initial_prompt,
        'stream': False
    }
    return _query_model(payload)

if __name__ == "__main__":
    embeddings_dict = dict()
    with open(initial_prompts_path) as f:
        data = json.load(f)
        
    for bug_id in data:
        initial_prompt = data[bug_id]
        embedding = get_LLM_response(initial_prompt)
        embeddings_dict[bug_id] = embedding
    
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings_dict, f)