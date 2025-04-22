import json
import requests
import time

endpoint = 'http://localhost:11434/api/embeddings'

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


def get_embeddings(initial_prompt):
    payload = {
        'model': 'nomic-embed-text',
        'prompt': initial_prompt,
        'stream': False
    }
    return _query_model(payload)