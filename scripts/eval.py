# scripts/eval.py
import argparse, os, json, pickle, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_decoder.dataset import SpeechDataset
from neural_decoder.model import GRUDecoder

def ctc_greedy_decode(logits, T_eff):
    """
    Greedy CTC decode with blank=0, collapse repeats, drop blanks.
    logits: (B,T,C) or (T,C); T_eff: effective time length (after stride/kernel).
    """
    if logits.ndim == 3:
        logits = logits[0]
    path = logits[:T_eff].argmax(dim=-1).cpu().numpy().tolist()
    decoded, prev = [], None
    for p in path:
        if p != 0 and p != prev:
            decoded.append(p)
        prev = p
    return np.array(decoded, dtype=np.int32)

def ctc_prefix_beam_search(log_probs, beam_size=20, blank=0):
    """
    Prefix beam search for CTC decoding with proper p_b / p_nb bookkeeping.
    
    Args:
        log_probs: (T, C) torch.Tensor or np.ndarray of log-probabilities over classes.
        beam_size: beam width.
        blank: index of the CTC blank symbol (default 0).
    
    Returns:
        Numpy array of decoded token IDs (without blanks).
    """
    # Ensure numpy array on CPU
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.detach().cpu().numpy()
    
    T, C = log_probs.shape
    
    # Beam represented as two dictionaries:
    # p_b[prefix]: log p(prefix, last is blank)
    # p_nb[prefix]: log p(prefix, last is non-blank)
    p_b = {(): 0.0}
    p_nb = {(): -np.inf}
    
    for t in range(T):
        p_b_new = {}
        p_nb_new = {}
        
        # Set of all prefixes currently in the beam
        prefixes = set(p_b.keys()) | set(p_nb.keys())
        
        for prefix in prefixes:
            pb = p_b.get(prefix, -np.inf)
            pnb = p_nb.get(prefix, -np.inf)
            
            # 1) Transition to blank: prefix stays the same
            p_blank = log_probs[t, blank]
            prev = p_b_new.get(prefix, -np.inf)
            p_b_new[prefix] = np.logaddexp(
                prev,
                np.logaddexp(pb + p_blank, pnb + p_blank),
            )
            
            # 2) Transitions to non-blank labels
            for c in range(C):
                if c == blank:
                    continue
                
                p_c = log_probs[t, c]
                
                if len(prefix) > 0 and prefix[-1] == c:
                    # If last label is the same as current, CTC collapses repeats
                    # So we update the same prefix (not create a new one)
                    # Only extend from blank to avoid double-counting
                    prev_pnb_new = p_nb_new.get(prefix, -np.inf)
                    p_nb_new[prefix] = np.logaddexp(prev_pnb_new, pb + p_c)
                else:
                    # New character: create new prefix
                    new_prefix = prefix + (c,)
                    prev_pnb_new = p_nb_new.get(new_prefix, -np.inf)
                    # Can come from both blank and non-blank
                    p_nb_new[new_prefix] = np.logaddexp(
                        prev_pnb_new,
                        np.logaddexp(pb + p_c, pnb + p_c),
                    )
        
        # Prune to top beam_size prefixes by total probability
        all_prefixes = set(p_b_new.keys()) | set(p_nb_new.keys())
        scores = []
        for prefix in all_prefixes:
            pb = p_b_new.get(prefix, -np.inf)
            pnb = p_nb_new.get(prefix, -np.inf)
            p_total = np.logaddexp(pb, pnb)
            scores.append((p_total, prefix))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:beam_size]
        
        # Reset beam to pruned set
        p_b = {}
        p_nb = {}
        for _, prefix in top:
            p_b[prefix] = p_b_new.get(prefix, -np.inf)
            p_nb[prefix] = p_nb_new.get(prefix, -np.inf)
    
    # Pick best prefix by total probability
    best_prefix = ()
    best_score = -np.inf
    for prefix in set(p_b.keys()) | set(p_nb.keys()):
        pb = p_b.get(prefix, -np.inf)
        pnb = p_nb.get(prefix, -np.inf)
        p_total = np.logaddexp(pb, pnb)
        if p_total > best_score:
            best_score = p_total
            best_prefix = prefix
    
    return np.array(best_prefix, dtype=np.int32)

def edit_distance(a, b):
    """Simple Levenshtein distance for small int sequences."""
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1, dtype=np.int32)
    dp[0, :] = np.arange(n + 1, dtype=np.int32)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])

def load_state_dict_flex(path, device):
    """Load a checkpoint that may be a state_dict, a dict with keys, or a whole Module."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        # common wrappers
        for k in ("state_dict", "model_state_dict", "net", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # raw state-dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelPath", required=True,
                        help="Folder containing 'args' and checkpoint (e.g., modelWeights)")
    parser.add_argument("--datasetPath", required=True,
                        help="Formatted pickle produced by format notebook (ptDecoder_ctc)")
    parser.add_argument("--split", choices=["test", "competition", "train"], default="test")
    parser.add_argument("--out", required=True, help="Path to write JSON metrics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_beam_search", action="store_true", help="Use prefix beam search instead of greedy decoding")
    parser.add_argument("--beam_size", type=int, default=20, help="Beam size for beam search (default: 20)")
    a = parser.parse_args()

    # 1) training hyperparams (to recreate model exactly)
    args_path = os.path.join(a.modelPath, "args")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Missing args file at {args_path}")
    with open(args_path, "rb") as fh:
        train_args = pickle.load(fh)

    # 2) formatted dataset
    with open(a.datasetPath, "rb") as fh:
        loadedData = pickle.load(fh)
    if a.split not in loadedData:
        raise RuntimeError(f"Split '{a.split}' not found in dataset. Available: {list(loadedData.keys())}")

    split_days = loadedData[a.split]
    test_ds = SpeechDataset(split_days)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # 3) model with correct shapes (per repo signature)
    model = GRUDecoder(
        neural_dim   = int(train_args["nInputFeatures"]),
        n_classes    = int(train_args["nClasses"]),
        hidden_dim   = int(train_args["nUnits"]),
        layer_dim    = int(train_args["nLayers"]),
        nDays        = int(train_args.get("nDays", 24)),
        dropout      = float(train_args.get("dropout", 0.0)),
        device       = a.device,
        strideLen    = int(train_args.get("strideLen", 4)),
        kernelLen    = int(train_args.get("kernelLen", 32)),
        gaussianSmoothWidth = float(train_args.get("gaussianSmoothWidth", 2.0)),
        bidirectional= bool(train_args.get("bidirectional", True)),
        use_layer_norm = bool(train_args.get("use_layer_norm", False)),
        input_dropout  = float(train_args.get("input_dropout", 0.0)),
    ).to(a.device)

    # 4) checkpoint (supports common names & no-extension files)
    preferred = [
        "best_model.pt", "model.pt", "checkpoint.pt", "final_model.pt",
        "modelWeights", "trainingState", "trainingStats"
    ]
    candidates = [os.path.join(a.modelPath, name) for name in preferred if os.path.isfile(os.path.join(a.modelPath, name))]
    # then any other file (except 'args')
    for p in sorted(glob.glob(os.path.join(a.modelPath, "*"))):
        base = os.path.basename(p)
        if os.path.isfile(p) and base != "args" and p not in candidates:
            candidates.append(p)

    state_dict = None
    picked = None
    for p in candidates:
        try:
            sd = load_state_dict_flex(p, a.device)
            if sd is not None:
                # strip DDP 'module.' prefix if present
                if any(k.startswith("module.") for k in sd.keys()):
                    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                model.load_state_dict(sd, strict=True)
                state_dict, picked = sd, p
                break
        except Exception:
            continue

    if state_dict is None:
        tried = ", ".join(os.path.basename(x) for x in candidates) or "(no files found)"
        raise FileNotFoundError(f"No usable checkpoint in {a.modelPath}. Checked: {tried}")
    print(f"[info] Loaded checkpoint: {picked}")

    # 5) evaluation loop (PER)
    model.eval()
    per_list = []
    sample_details = []

    with torch.no_grad():
        for X, y, X_len, y_len, dayIdx in test_loader:
            X      = X.to(a.device)
            y      = y.to(a.device)
            X_len  = X_len.to(a.device)
            y_len  = y_len.to(a.device)
            dayIdx = dayIdx.to(a.device)

            logits = model(X, dayIdx)  # (B, T', C) with CTC blank added internally
            T_eff = ((X_len - model.kernelLen) // model.strideLen).to(torch.int32)
            T_eff_int = int(T_eff[0].item())

            if a.use_beam_search:
                # Use prefix beam search
                log_probs = logits[0, :T_eff_int].log_softmax(dim=-1)
                hyp = ctc_prefix_beam_search(log_probs, beam_size=a.beam_size, blank=0)
            else:
                # Use greedy decoding
                hyp = ctc_greedy_decode(logits[0], T_eff_int)
            
            tgt = y[0, : y_len[0]].cpu().numpy()

            dist = edit_distance(hyp, tgt)
            per = dist / max(1, len(tgt))
            per_list.append(per)
            if len(sample_details) < 50:  # keep JSON small
                sample_details.append({
                    "len_target": int(len(tgt)),
                    "len_hyp": int(len(hyp)),
                    "distance": int(dist),
                    "per": float(per),
                })

    metrics = {
        "split": a.split,
        "num_samples": len(per_list),
        "avg_PER": float(np.mean(per_list) if per_list else float("nan")),
        "median_PER": float(np.median(per_list) if per_list else float("nan")),
        "sample_details": sample_details,
        "checkpoint": os.path.basename(picked) if picked else None,
        "decoding_method": "beam_search" if a.use_beam_search else "greedy",
        "beam_size": a.beam_size if a.use_beam_search else None,
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {a.out}")
    print(f"Avg PER: {metrics['avg_PER']:.4f} over {metrics['num_samples']} samples")

if __name__ == "__main__":
    main()
