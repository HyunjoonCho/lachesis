from routers.routers import MatrixFactorizationRouter
import json
import numpy as np
import tqdm

ck_path = 'routers/matrix_factorization/best_checkpoint.pt'
prompt_path = 'route_data/initial_prompts.json'
probs_path = 'results/no_cv.npy'

if __name__ == "__main__":
    router = MatrixFactorizationRouter(ck_path, strong_model='gpt-4o', weak_model='llama3-8b')
    with open(prompt_path) as f:
        prompts = json.load(f)
    
    win_rates = list()
    for bug_id in tqdm.tqdm(prompts):
        prompt = prompts[bug_id]
        win_rates.append(router.calculate_strong_win_rate(prompt))
    
    np.save(probs_path, np.array(win_rates))