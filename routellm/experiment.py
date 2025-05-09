from routers.routers import MatrixFactorizationRouter
import json
import torch
import os
import numpy as np
import random
from sklearn.model_selection import KFold
from routers.matrix_factorization.train_matrix_factorization import *

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

checkpoint_dir = 'routers/matrix_factorization/cv'
probs_dir = 'results/cv'
prompt_path = 'route_data/initial_prompts.json'
combined_result_path = '../AutoFL/combined_fl_results/d4j_gpt4o_results_R10_full.json'

if __name__ == "__main__":
    strong_model_name = 'gpt-4o'
    weak_model_name = 'llama3-8b'

    json_path = f'route_data/{strong_model_name}_vs_{weak_model_name}.json'
    npy_path = f'route_data/{strong_model_name}_{weak_model_name}_embeddings.npy'

    dim = 128
    batch_size = 16
    num_epochs = 30
    alpha = 0.1
    use_proj = True
    lr = 3e-4
    weight_decay = 1e-5
    k = 5

    data = json.load(open(json_path, "r"))

    filtered_data = [
        sample
        for sample in data
        if sample["winner"] in ["model_a", "model_b"]
        and sample["model_a"] != sample["model_b"]
    ]

    with open(prompt_path) as f:
        prompts = json.load(f)

    with open(combined_result_path) as f:
        combined_bug_indices = list(json.load(f)['buggy_methods'].keys())

    random.shuffle(filtered_data)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(filtered_data)):
        print(f"\nFold {fold + 1}/{k}")

        all_train_data = [filtered_data[i] for i in train_idx]
        test_data = [filtered_data[i] for i in test_idx]

        train_size = int(0.75 * len(all_train_data))
        
        train_data = all_train_data[:train_size]
        val_data = all_train_data[train_size:]
        
        print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

        train_loader = PairwiseDataset(train_data).get_dataloaders(
            batch_size=batch_size, shuffle=True
        )
        val_loader = PairwiseDataset(val_data).get_dataloaders(
            batch_size=8, shuffle=False
        )

        model = MFModel_Train(
            dim=dim,
            num_models=len(MODEL_IDS),
            num_prompts=len(data),
            use_proj=use_proj,
            npy_path=npy_path,
        ).to("cuda")
        
        os.makedirs(f"{checkpoint_dir}/{strong_model_name}_{weak_model_name}", exist_ok=True) 
        save_path = f"{checkpoint_dir}/{strong_model_name}_{weak_model_name}/best_{fold + 1}.pt"
        train_loops(
            model,
            train_loader,
            val_loader,
            lr=lr,
            weight_decay=weight_decay,
            alpha=alpha,
            num_epochs=num_epochs,
            device="cuda",
            save_path=save_path,
        )

        router = MatrixFactorizationRouter(
            save_path, 
            strong_model=strong_model_name, 
            weak_model=weak_model_name
        )
        
        test_bug_ids = [combined_bug_indices[i] for i in test_idx]
        win_rates = dict()
        for bug_id in test_bug_ids:
            prompt = prompts[bug_id]
            win_rates[bug_id] = router.calculate_strong_win_rate(prompt)

        os.makedirs(f'{probs_dir}/{strong_model_name}_{weak_model_name}', exist_ok=True)
        with open(f'{probs_dir}/{strong_model_name}_{weak_model_name}/fold_{fold + 1}.json', 'w') as f:
            json.dump(win_rates, f)
