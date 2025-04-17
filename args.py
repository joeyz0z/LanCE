import argparse


def get_args():
    parser = argparse.ArgumentParser(description="domain-invariant CBM.")

    # HYPERPARAMETERS ##
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', action="store_true", default=False)
    # parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--batch_size', type=str, default=256)
    parser.add_argument("--alpha", type=float, default=1, help="orthogonality loss weight")
    parser.add_argument("--beta", type=float, default=0, help="concept supervision weight")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=float, default=100, help="epochs")

    ## Setting ##
    parser.add_argument('--class_avg_concept', action="store_true", default=False)  # CUB, RIVAL10

    ## DATA/MODEL PATH ##
    parser.add_argument('--dataset', type=str, default='CUB')  # CUB, AWA2, LADA, LADV
    parser.add_argument('--data_dir', type=str, default='./data')  # CUB, AWA2, LADA, LADV
    parser.add_argument('--CLIP_type', type=str, default='ViT-L/14') # ViT-B/32, ViT-L/14
    parser.add_argument('--CBM_type', type=str, default='clip_cbm') # clip_cbm, cliplp
    parser.add_argument('--model_save_path', type=str, default=r'logs')
    parser.add_argument('--prompt_type', type=str, default=r'origin') # origin, others
    parser.add_argument("--save_model", action="store_true", default=True)
    args = parser.parse_args()

    return args