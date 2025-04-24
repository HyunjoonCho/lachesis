from routers.routers import MatrixFactorizationRouter
import json
import tqdm

import torch
import numpy as np
import random
from sklearn.model_selection import KFold

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

checkpoint_dir = 'routers/matrix_factorization/cv'
probs_dir = 'results/cv'
prompt_path = 'route_data/initial_prompts.json'
json_path = "route_data/vs.json"
gpt_result_path = '../AutoFL/combined_fl_results/d4j_gpt4o_results_R10_full.json'

if __name__ == "__main__":
    with open(prompt_path) as f:
        prompts = json.load(f)

    with open(gpt_result_path) as f:
        gpt_bug_indices = list(json.load(f)['buggy_methods'].keys())

    with open(json_path) as f:
        data = json.load(f)

    filtered_data = [
        sample
        for sample in data
        if sample["winner"] in ["model_a", "model_b"]
        and sample["model_a"] != sample["model_b"]
    ]

    random.shuffle(filtered_data)
    k = 5 # identical to the training setting
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (_, test_idx) in enumerate(kf.split(filtered_data)):
        test_bug_ids = [gpt_bug_indices[i] for i in test_idx]
        router = MatrixFactorizationRouter(f'{checkpoint_dir}/best_{fold + 1}.pt', strong_model='gpt-4o', weak_model='llama3-8b')
        win_rates = dict()
        for bug_id in tqdm.tqdm(test_bug_ids):
            prompt = prompts[bug_id]
            win_rates[bug_id] = router.calculate_strong_win_rate(prompt)
        
        with open(f'{probs_dir}/fold_{fold + 1}.json', 'w') as f:
            json.dump(win_rates, f)