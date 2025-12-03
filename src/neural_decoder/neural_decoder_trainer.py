import os
import pickle
import time
import json

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
    args=None,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    # Get time masking parameters from args if available
    if args is None:
        args = {}
    time_mask_prob = args.get("time_mask_prob", 0.10)
    time_mask_width = args.get("time_mask_width", 20)
    time_mask_max_masks = args.get("time_mask_max_masks", 1)
    
    train_ds = SpeechDataset(
        loadedData["train"], 
        transform=None, 
        split="train",
        time_mask_prob=time_mask_prob,
        time_mask_width=time_mask_width,
        time_mask_max_masks=time_mask_max_masks
    )
    test_ds = SpeechDataset(loadedData["test"], split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        args,
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        use_layer_norm=args.get("use_layer_norm", False),
        input_dropout=args.get("input_dropout", 0.0),
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    # Use AdamW optimizer with weight_decay parameter
    optimizer_name = args.get("optimizer", "adamw").lower()
    weight_decay = args.get("weight_decay", args.get("l2_decay", 1e-4))
    
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["lrStart"],
            weight_decay=weight_decay,
        )
    else:
        # Fallback to Adam for backward compatibility
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=weight_decay,
        )
    
    # Run #3: Linear warmup → cosine decay
    warmup_steps = args.get("warmup_steps", 1000)
    cosine_steps = args.get("cosine_T_max", args["nBatch"] - warmup_steps)
    
    # Use LambdaLR for warmup (more reliable than LinearLR with SequentialLR)
    # Warmup: linearly increase from 0 to lrStart over warmup_steps
    def warmup_lambda(step):
        if step >= warmup_steps:
            return 1.0
        return float(step) / float(max(1, warmup_steps))
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_lambda,
    )
    
    # Cosine decay: decay from lrStart to lrEnd over remaining steps
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=args["lrEnd"],
    )
    
    # Chain schedulers: warmup first, then cosine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    
    # Run #3: Track moving average CER (over last 200 steps)
    moving_avg_window = 200
    cer_history = []
    best_cer_ma = float('inf')
    
    # Metrics logging file
    metrics_file = os.path.join(args["outputDir"], "metrics.jsonl")
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Run #3: Strong gradient clipping to prevent loss spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Step scheduler after optimizer

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                # Run #3: Update moving average CER
                cer_history.append(cer)
                if len(cer_history) > moving_avg_window:
                    cer_history.pop(0)
                cer_ma = np.mean(cer_history) if cer_history else cer

                endTime = time.time()
                time_per_batch = (endTime - startTime) / 100
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, cer_ma: {cer_ma:>7f}, time/batch: {time_per_batch:>7.3f}, lr: {current_lr:.6f}"
                )
                
                # Log metrics to JSONL file
                metrics_entry = {
                    "step": batch,
                    "ctc_loss": float(avgDayLoss),
                    "cer": float(cer),
                    "cer_ma": float(cer_ma),
                    "lr": float(current_lr),
                    "time_per_batch": float(time_per_batch),
                }
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics_entry) + "\n")
                
                startTime = time.time()

            # Run #3: Save best checkpoint based on moving average CER
            if cer_ma < best_cer_ma:
                best_cer_ma = cer_ma
                # Save best model checkpoint
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                torch.save(model.state_dict(), os.path.join(args["outputDir"], "best_model.pt"))
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)
    
    # Save final model checkpoint
    torch.save(model.state_dict(), os.path.join(args["outputDir"], "final_model.pt"))


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        use_layer_norm=args.get("use_layer_norm", False),
        input_dropout=args.get("input_dropout", 0.0),
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()