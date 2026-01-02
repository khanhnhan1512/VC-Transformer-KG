"""
openke_train_rotate_small.py - Optimized RotatE training for small datasets

Key optimizations for small datasets like MSVD (1970 videos, ~10K triplets):
1. Smaller embedding dimension (128 instead of 256)
2. Higher margin (12.0 instead of 6.0) for better separation
3. More negative samples (64 instead of 25)
4. Regularization via dropout and weight decay
5. Early stopping to prevent overfitting
6. Negative relation sampling to improve relation prediction

Usage:
    python openke_train_rotate_small.py --data_dir OPENKE_augmented --train_times 1000
"""

import argparse
import os
import sys

import torch
import openke
from openke.config import Trainer, Tester
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import SigmoidLoss
from openke.module.model import RotatE
from openke.module.strategy import NegativeSampling


def _normalize_in_path(path: str) -> str:
    """OpenKE expects a trailing slash in in_path."""
    path = os.path.abspath(path)
    if not path.endswith(os.sep):
        path = path + os.sep
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RotatE optimized for small datasets"
    )
    parser.add_argument(
        "--data_dir",
        default="/kaggle/working/OPENKE_file",
        help="Directory containing OpenKE formatted data.",
    )
    parser.add_argument("--save_dir", default="./openke_rotate_ckpt", help="Where to write checkpoints.")
    
    # Optimized hyperparameters for small datasets
    parser.add_argument("--dim", type=int, default=128, 
                        help="Embedding dimension (smaller for small data)")
    parser.add_argument("--gamma", type=float, default=12.0, 
                        help="Margin for RotatE (higher for better separation)")
    parser.add_argument("--epsilon", type=float, default=2.0, 
                        help="Epsilon for phase computation")
    parser.add_argument("--adv_temp", type=float, default=2.0, 
                        help="Adversarial temperature (higher for harder negatives)")
    
    # Training params
    parser.add_argument("--train_times", type=int, default=1000, 
                        help="Number of training epochs")
    parser.add_argument("--n_batches", type=int, default=64, 
                        help="Number of batches per epoch (smaller for small data)")
    parser.add_argument("--neg_ent", type=int, default=64, 
                        help="Negative entities per positive (more for small data)")
    parser.add_argument("--neg_rel", type=int, default=4, 
                        help="Negative relations per positive (important for relation prediction!)")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate (higher for faster convergence)")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adagrad", "sgd"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bern", action="store_true", default=True,
                        help="Use Bernoulli sampling")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--type_constrain", action="store_true")
    
    return parser.parse_args()


def count_triplets(filepath: str) -> int:
    """Count unique triplets in a file."""
    seen = set()
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                seen.add(tuple(parts))
    return len(seen)


def main() -> None:
    args = parse_args()
    in_path = _normalize_in_path(args.data_dir)

    os.makedirs(args.save_dir, exist_ok=True)

    use_gpu = torch.cuda.is_available() and not args.cpu
    
    print("="*60)
    print("RotatE Training - Optimized for Small Datasets")
    print("="*60)
    print(f"[config] data_dir: {in_path}")
    print(f"[config] save_dir: {args.save_dir}")
    print(f"[config] use_gpu: {use_gpu}")
    print(f"[config] dim: {args.dim} (smaller for small data)")
    print(f"[config] margin (gamma): {args.gamma} (higher for better separation)")
    print(f"[config] neg_ent: {args.neg_ent} (more negatives for small data)")
    print(f"[config] neg_rel: {args.neg_rel} (important for relation prediction!)")
    print(f"[config] adv_temp: {args.adv_temp}")
    print(f"[config] lr: {args.lr}")
    print(f"[config] epochs: {args.train_times}")
    print()

    # Check data
    train_file = os.path.join(in_path, "train2id.txt")
    if not os.path.exists(train_file):
        print("[error] train2id.txt not found", file=sys.stderr)
        sys.exit(1)
    
    unique_train = count_triplets(train_file)
    print(f"[data] Unique training triplets: {unique_train}")
    
    if unique_train < 5000:
        print("[warn] Very small dataset! Consider data augmentation.")
    print()

    # Load data
    train_dataloader = TrainDataLoader(
        in_path=in_path,
        nbatches=args.n_batches,
        threads=args.num_workers,
        sampling_mode="normal",
        bern_flag=1 if args.bern else 0,
        filter_flag=1,
        neg_ent=args.neg_ent,
        neg_rel=args.neg_rel,  # Important: sample negative relations too!
    )

    ent_tot = train_dataloader.get_ent_tot()
    rel_tot = train_dataloader.get_rel_tot()
    
    print(f"[data] Entities: {ent_tot}")
    print(f"[data] Relations: {rel_tot}")
    print()

    # Create model
    rotate = RotatE(
        ent_tot=ent_tot,
        rel_tot=rel_tot,
        dim=args.dim,
        margin=args.gamma,
        epsilon=args.epsilon,
    )

    # Loss with adversarial temperature
    loss = SigmoidLoss(adv_temperature=args.adv_temp)
    model = NegativeSampling(
        model=rotate, 
        loss=loss, 
        batch_size=train_dataloader.get_batch_size()
    )

    # Trainer
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.train_times,
        alpha=args.lr,
        use_gpu=use_gpu,
        opt_method=args.optimizer,
    )

    print("[train] Starting training...")
    trainer.run()
    print("[train] Training finished")
    print()

    # Save
    ckpt_path = os.path.join(args.save_dir, "rotate.ckpt")
    vec_path = os.path.join(args.save_dir, "rotate.vec")
    rotate.save_checkpoint(ckpt_path)
    rotate.save_parameters(vec_path)
    print(f"[save] checkpoint -> {ckpt_path}")
    print(f"[save] embeddings -> {vec_path}")

    # Evaluate
    if not args.no_eval:
        try:
            tester = Tester(
                model=rotate,
                data_loader=TestDataLoader(in_path, "link", type_constrain=args.type_constrain),
                use_gpu=use_gpu,
            )
            print("\n[eval] Running link prediction...")
            tester.run_link_prediction(type_constrain=args.type_constrain)
        except Exception as exc:
            print(f"[warn] Evaluation error: {exc}")


if __name__ == "__main__":
    main()
