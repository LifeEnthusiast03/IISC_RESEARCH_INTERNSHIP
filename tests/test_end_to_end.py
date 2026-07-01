import os
import sys
import json
import numpy as np
import joblib

# Critical Windows DLL fix: import sklearn and xgboost before torch
import sklearn
import xgboost as xgb
import torch

# Add training to path so we can import models
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "training"))

from train_autoencoder import Autoencoder, INPUT_DIM
from train_attack_type_nn import AttackTypeNN
from train_dqn import DQNNetwork

def main():
    print("="*70)
    print("  END-TO-END EVALUATION PIPELINE (Attack Data Only)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Data
    data_dir = os.path.join(_PROJECT_ROOT, "data", "processed")
    model_dir = os.path.join(_PROJECT_ROOT, "models")
    
    print("\n[Loading Attack Data...]")
    X_attacks = np.load(os.path.join(data_dir, "X_attacks.npy"))
    y_attacks = np.load(os.path.join(data_dir, "y_attacks_str.npy"), allow_pickle=True)
    total_attacks = len(X_attacks)
    print(f"  Loaded {total_attacks:,} attack flows.")
    
    unique_attacks = np.unique(y_attacks)
    
    # 2. Load Models
    print("\n[Loading Models...]")
    
    # Autoencoder
    ae = Autoencoder(input_dim=INPUT_DIM).to(device)
    ae.load_state_dict(torch.load(os.path.join(model_dir, "autoencoder.pt"), map_location=device, weights_only=True))
    ae.eval()
    
    with open(os.path.join(model_dir, "threshold.json"), "r") as f:
        threshold = json.load(f)["threshold"]
    print(f"  [✓] Autoencoder loaded (Threshold: {threshold:.6f})")
    
    # Hybrid Classifier
    hybrid_clf = joblib.load(os.path.join(model_dir, "hybrid_classifier.pkl"))
    hybrid_encoder = joblib.load(os.path.join(model_dir, "hybrid_label_encoder.pkl"))
    print("  [✓] Hybrid Classifier loaded")
    
    # Attack Type NN
    with open(os.path.join(data_dir, "attack_type_label_map.json"), "r") as f:
        at_label_map_raw = json.load(f)
    at_label_map = {int(k): v for k, v in at_label_map_raw.items()}
    n_at_classes = len(at_label_map)
    
    atnn = AttackTypeNN(input_dim=INPUT_DIM, n_classes=n_at_classes).to(device)
    atnn.load_state_dict(torch.load(os.path.join(model_dir, "attack_type_nn.pt"), map_location=device, weights_only=True))
    atnn.eval()
    print("  [✓] Attack-Type NN loaded")
    
    # DQN
    state_dim = INPUT_DIM + 1 + n_at_classes + 1
    dqn = DQNNetwork(state_dim=state_dim, n_actions=5).to(device)
    dqn.load_state_dict(torch.load(os.path.join(model_dir, "dqn_agent.pt"), map_location=device, weights_only=True))
    dqn.eval()
    print("  [✓] DQN Agent loaded")
    
    # Prepare Tensor for PyTorch models
    X_tensor = torch.tensor(X_attacks, dtype=torch.float32, device=device)
    
    # 3. Level 1: Autoencoder
    print("\n" + "-"*70)
    print("LEVEL 1: AUTOENCODER (Anomaly Detection)")
    print("-"*70)
    
    batch_size = 4096
    recon_errors = []
    
    with torch.no_grad():
        for i in range(0, total_attacks, batch_size):
            batch = X_tensor[i:i+batch_size]
            recon = ae(batch)
            err = ((batch - recon) ** 2).mean(dim=1).cpu().numpy()
            recon_errors.append(err)
            
    reconstruction_errors = np.concatenate(recon_errors)
    
    is_anomaly_ae = reconstruction_errors > threshold
    ae_caught = np.sum(is_anomaly_ae)
    print(f"  Overall Attacks flagged as Anomaly: {ae_caught:,} / {total_attacks:,} ({ae_caught/total_attacks*100:.2f}%)\n")
    
    print("  Per-class breakdown (Autoencoder caught vs total):")
    for attack_type in unique_attacks:
        mask = (y_attacks == attack_type)
        total_type = np.sum(mask)
        caught_type = np.sum(is_anomaly_ae[mask])
        print(f"    - {attack_type:<20}: {caught_type:>8,} / {total_type:>8,} ({caught_type/total_type*100:>5.2f}%)")
    
    # 4. Level 2: Autoencoder + Hybrid Classifier
    print("\n" + "-"*70)
    print("LEVEL 2: HYBRID CLASSIFIER (Defense-in-Depth for Missed Attacks)")
    print("-"*70)
    
    missed_by_ae_idx = np.where(~is_anomaly_ae)[0]
    hybrid_caught = 0
    is_hybrid_caught = np.zeros(len(missed_by_ae_idx), dtype=bool)
    
    if len(missed_by_ae_idx) > 0:
        X_missed = X_attacks[missed_by_ae_idx]
        y_attacks_missed = y_attacks[missed_by_ae_idx]
        
        # XGBoost prediction
        y_hybrid_pred_idx = hybrid_clf.predict(X_missed)
        y_hybrid_pred = np.array([hybrid_encoder[i] for i in y_hybrid_pred_idx])
        
        # Any prediction != Benign means the hybrid classifier caught the attack
        is_hybrid_caught = (y_hybrid_pred != "Benign")
        hybrid_caught = np.sum(is_hybrid_caught)
        print(f"  Overall attacks missed by AE      : {len(missed_by_ae_idx):,}")
        print(f"  Overall caught by Hybrid Classifier: {hybrid_caught:,} ({hybrid_caught/max(1, len(missed_by_ae_idx))*100:.2f}%)\n")
        
        print("  Per-class breakdown (Hybrid Classifier caught vs missed by AE):")
        missed_types = np.unique(y_attacks_missed)
        for attack_type in missed_types:
            mask = (y_attacks_missed == attack_type)
            total_type = np.sum(mask)
            caught_type = np.sum(is_hybrid_caught[mask])
            print(f"    - {attack_type:<20}: {caught_type:>8,} / {total_type:>8,} ({caught_type/max(1, total_type)*100:>5.2f}%)")
    else:
        print("  No attacks missed by AE to pass to Hybrid Classifier.")
        
    total_is_caught = is_anomaly_ae.copy()
    if len(missed_by_ae_idx) > 0:
        total_is_caught[missed_by_ae_idx] = is_hybrid_caught

    total_caught = np.sum(total_is_caught)
    print("\n  ================================================================")
    print(f"  OVERALL PROPERLY CLASSIFIED (AE + Hybrid): {total_caught:,} / {total_attacks:,} ({total_caught/total_attacks*100:.2f}%)")
    print("  ================================================================")
    print("  Overall Per-class breakdown (AE + Hybrid caught vs total):")
    for attack_type in unique_attacks:
        mask = (y_attacks == attack_type)
        total_type = np.sum(mask)
        caught_type = np.sum(total_is_caught[mask])
        print(f"    - {attack_type:<20}: {caught_type:>8,} / {total_type:>8,} ({caught_type/total_type*100:>5.2f}%)")
    
    # 5. Level 3: Attack-Type NN
    print("\n" + "-"*70)
    print("LEVEL 3: ATTACK-TYPE NN (Multi-Class Classification)")
    print("-"*70)
    
    atnn_preds = []
    atnn_probs = []
    atnn_conf = []
    
    with torch.no_grad():
        for i in range(0, total_attacks, batch_size):
            batch = X_tensor[i:i+batch_size]
            logits = atnn(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf = probs.max(dim=1, keepdim=True).values
            
            atnn_probs.append(probs.cpu().numpy())
            atnn_conf.append(conf.cpu().numpy())
            atnn_preds.append(logits.argmax(dim=1).cpu().numpy())
            
    atnn_probs = np.concatenate(atnn_probs)
    atnn_conf = np.concatenate(atnn_conf)
    y_atnn_pred_idx = np.concatenate(atnn_preds)
    
    y_atnn_pred_str = np.array([at_label_map.get(i, "Unknown") for i in y_atnn_pred_idx])
    
    is_atnn_correct = (y_atnn_pred_str == y_attacks)
    atnn_correct = np.sum(is_atnn_correct)
    print(f"  Overall exact match accuracy: {atnn_correct:,} / {total_attacks:,} ({atnn_correct/total_attacks*100:.2f}%)\n")
    
    print("  Per-class breakdown (Attack-Type NN exact match vs total):")
    for attack_type in unique_attacks:
        mask = (y_attacks == attack_type)
        total_type = np.sum(mask)
        correct_type = np.sum(is_atnn_correct[mask])
        print(f"    - {attack_type:<20}: {correct_type:>8,} / {total_type:>8,} ({correct_type/max(1, total_type)*100:>5.2f}%)")
    
    # 6. Level 4: DQN Agent
    print("\n" + "-"*70)
    print("LEVEL 4: DQN AGENT (Remediation Suggestions)")
    print("-"*70)
    
    ae_err_np = reconstruction_errors.reshape(-1, 1)
    states = np.concatenate([X_attacks, ae_err_np, atnn_probs, atnn_conf], axis=1)
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    
    dqn_actions = []
    with torch.no_grad():
        for i in range(0, total_attacks, batch_size):
            batch = states_tensor[i:i+batch_size]
            q_values = dqn(batch)
            dqn_actions.append(q_values.argmax(dim=1).cpu().numpy())
            
    best_actions = np.concatenate(dqn_actions)
    
    action_names = {
        0: "Block IP",
        1: "Revoke Credentials",
        2: "Isolate Server",
        3: "Kill Process",
        4: "Monitor",
    }
    
    print("  Overall DQN suggested actions distribution:")
    unique, counts = np.unique(best_actions, return_counts=True)
    for u, c in zip(unique, counts):
        action_name = action_names.get(u, "Unknown")
        print(f"    - {action_name:<20} (Action {u}): {c:>8,} flows ({c/total_attacks*100:>5.2f}%)")
        
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
