import argparse
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from model_torch import Generator, Discriminator
from dataset_torch import get_dataloader

if __name__ == "__main__":

    # ── Args ────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lambda_l1",  type=int,   default=100)
    parser.add_argument("--student_id", type=str,   default="202200622")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    train_loader = get_dataloader(
        "data/sketch2pokemon/trainA",
        "data/sketch2pokemon/trainB",
        batch_size=args.batch_size,
    )
    print(f"Dataset loaded: {len(train_loader)} batches")

    # ── Models ───────────────────────────────────────────────────────────────
    print("Building models...")
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print("Models ready")

    # ── MLflow ───────────────────────────────────────────────────────────────
    print("Starting MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Assignment3_Salma")

    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": args.lr,
            "batch_size":    args.batch_size,
            "epochs":        args.epochs,
            "lambda_l1":     args.lambda_l1,
        })
        mlflow.set_tag("student_id", args.student_id)
        print("MLflow run started")

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(1, args.epochs + 1):
            epoch_g, epoch_d, epoch_l1 = 0.0, 0.0, 0.0

            for inp, tar in train_loader:
                inp, tar = inp.to(DEVICE), tar.to(DEVICE)

                fake   = G(inp)
                d_real = D(inp, tar)
                d_fake = D(inp, fake.detach())
                d_loss = (
                    bce(d_real, torch.ones_like(d_real)) +
                    bce(d_fake, torch.zeros_like(d_fake))
                )
                d_opt.zero_grad(); d_loss.backward(); d_opt.step()

                d_fake = D(inp, fake)
                g_gan  = bce(d_fake, torch.ones_like(d_fake))
                g_l1   = l1(fake, tar) * args.lambda_l1
                g_loss = g_gan + g_l1
                g_opt.zero_grad(); g_loss.backward(); g_opt.step()

                epoch_g  += g_loss.item()
                epoch_d  += d_loss.item()
                epoch_l1 += g_l1.item() / args.lambda_l1

            n = len(train_loader)
            mlflow.log_metrics({
                "G_loss": epoch_g  / n,
                "D_loss": epoch_d  / n,
                "L1":     epoch_l1 / n,
            }, step=epoch)

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"G: {epoch_g/n:.3f}  D: {epoch_d/n:.3f}  L1: {epoch_l1/n:.3f}")

        mlflow.pytorch.log_model(G, artifact_path="generator")
        print("Model saved to MLflow.")