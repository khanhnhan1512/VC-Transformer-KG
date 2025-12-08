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
    """OpenKE expects a trailing slash in in_path; normalize to avoid common path bugs."""
    path = os.path.abspath(path)
    if not path.endswith(os.sep):
        path = path + os.sep
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RotatE model on the provided OpenKE-formatted KG triples."
    )
    parser.add_argument(
        "--data_dir",
        default="/kaggle/working/OPENKE_file",
        help="Directory containing train2id.txt, valid2id.txt, test2id.txt, entity2id.txt, relation2id.txt.",
    )
    parser.add_argument("--save_dir", default="./openke_rotate_ckpt", help="Where to write checkpoints/embeddings.")
    parser.add_argument("--result_dir", default="./openke_rotate_result", help="Where to write evaluation logs.")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension for RotatE.")
    parser.add_argument("--gamma", type=float, default=6.0, help="Margin (gamma) for RotatE.")
    parser.add_argument("--epsilon", type=float, default=2.0, help="Small epsilon used in RotatE phase computation.")
    parser.add_argument("--adv_temp", type=float, default=1.0, help="Adversarial temperature for Sigmoid loss.")
    parser.add_argument("--train_times", type=int, default=800, help="Number of training epochs.")
    parser.add_argument("--n_batches", type=int, default=128, help="Number of batches per epoch.")
    parser.add_argument("--neg_ent", type=int, default=25, help="Negative entities per positive triple.")
    parser.add_argument("--neg_rel", type=int, default=0, help="Negative relations per positive triple.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "adagrad", "sgd"],
        help="Optimizer used by OpenKE trainer.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Data loader threads.")
    parser.add_argument("--bern", action="store_true", help="Enable Bernoulli trick for sampling mode.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--no_eval", action="store_true", help="Skip link prediction/triple classification eval after training.")
    parser.add_argument(
        "--type_constrain",
        action="store_true",
        help="Use type_constrain.txt if available (otherwise leave off to avoid file requirement).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = _normalize_in_path(args.data_dir)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    use_gpu = torch.cuda.is_available() and not args.cpu
    print(f"[setup] data_dir={in_path} | save_dir={args.save_dir} | use_gpu={use_gpu}")

    if not os.path.exists(os.path.join(in_path, "train2id.txt")):
        print("[error] train2id.txt not found in data_dir", file=sys.stderr)
        sys.exit(1)

    train_dataloader = TrainDataLoader(
        in_path=in_path,
        nbatches=args.n_batches,
        threads=args.num_workers,
        sampling_mode="normal",
        bern_flag=1 if args.bern else 0,
        filter_flag=1,
        neg_ent=args.neg_ent,
        neg_rel=args.neg_rel,
    )

    rotate = RotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.gamma,
        epsilon=args.epsilon,
    )

    loss = SigmoidLoss(adv_temperature=args.adv_temp)
    model = NegativeSampling(model=rotate, loss=loss, batch_size=train_dataloader.get_batch_size())

    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.train_times,
        alpha=args.lr,
        use_gpu=use_gpu,
        opt_method=args.optimizer,
    )

    print("[train] starting training...")
    trainer.run()
    print("[train] training finished")

    ckpt_path = os.path.join(args.save_dir, "rotate.ckpt")
    vec_path = os.path.join(args.save_dir, "rotate.vec")
    rotate.save_checkpoint(ckpt_path)
    rotate.save_parameters(vec_path)
    print(f"[save] checkpoint -> {ckpt_path}")
    print(f"[save] embeddings -> {vec_path}")

    if not args.no_eval:
        try:
            tester = Tester(
                model=rotate,
                data_loader=TestDataLoader(in_path, "link", type_constrain=args.type_constrain),
                use_gpu=use_gpu,
            )
            print("[eval] running link prediction (may skip type_constrain if file missing)...")
            tester.run_link_prediction(type_constrain=args.type_constrain)
            print("[eval] running triple classification...")
            tester.run_triple_classification()
        except Exception as exc:  # pragma: no cover - runtime guard for optional eval
            print(f"[warn] evaluation skipped due to error: {exc}")


if __name__ == "__main__":
    main()
