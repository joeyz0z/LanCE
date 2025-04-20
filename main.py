import os
import time
import json
import logging
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
os.environ["WANDB_MODE"] = "offline"

# Local imports
from args import get_args
from model.cbm_models import clip_cbm_orth, clip_mlp
from data import get_dataset_classes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.args.device = self.device
        self._init_components()

        self.best_metrics = {
            'source_acc': 0.0,
            'target_acc': 0.0,
            'epoch': -1
        }

    def _setup_device(self):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available()  else "cpu"
        )
        logger.info(f"Using device: {device}")
        return device

    def _init_components(self):
        self._prepare_datasets()
        self._init_model()
        self._init_optimizer()
        self._init_loss_fns()

    def _prepare_datasets(self):
        """CUB, AWA2, LADA, LADV"""
        self.train_dataset, self.train_loader, self.source_test_dataset, self.source_test_loader, self.target_test_dataset, self.target_test_loader = get_dataset_classes(self.args)

        logger.info(f"Dataset {self.args.dataset} loaded")
        logger.info(f"Train samples: {len(self.train_dataset):,}")
        logger.info(f"Source test samples: {len(self.source_test_dataset):,}")
        logger.info(f"Target test samples: {len(self.target_test_dataset):,}")

    def _init_model(self):
        """clip_cbm, cliplp"""
        model_factory = {
            'clip_cbm': clip_cbm_orth,
            'cliplp': clip_mlp
        }

        try:
            model_class = model_factory[self.args.CBM_type]
        except KeyError:
            raise ValueError(f"Unsupported model type: {self.args.CBM_type}")

        self.model = model_class(
            args=self.args,
            class_names=list(self.train_dataset.classname2id.keys()),
            concept_names=list(self.train_dataset.concept2id.keys()),
            domain_diffs=self.train_dataset.domain_diffs
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {self.args.CBM_type}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_optimizer(self):
        """init optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        logger.info(f"Optimizer initialized with lr={self.args.lr:.1e}")

    def _init_loss_fns(self):
        """init loss"""
        self.loss_fns = {
            'cls': nn.CrossEntropyLoss(),
            'concept': nn.BCEWithLogitsLoss()
        }
        logger.info("Loss functions initialized")

    def _save_concept_logs(self):
        """save concept logs"""
        os.makedirs("logs", exist_ok=True)
        base_path = f"logs/{self.args.dataset}_{self.args.CBM_type}"

        last_concepts, best_concepts = self.model.extract_cls_concept()

        with open(f"{base_path}_last.json", 'w') as f:
            json.dump(last_concepts, f, indent=2)

        with open(f"{base_path}_best.json", 'w') as f:
            json.dump(best_concepts, f, indent=2)

        logger.info(f"Concept analysis saved to {base_path}_*.json")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.args.epochs}",
            dynamic_ncols=True
        )

        for batch_idx, (images, labels, attr_labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            attr_labels = attr_labels.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)

            concept_preds, cls_preds, reg_loss = self.model(images)

            cls_loss = self.loss_fns['cls'](cls_preds, labels)
            concept_loss = self.loss_fns['concept'](concept_preds, attr_labels)
            orth_loss = torch.abs(reg_loss).mean() if reg_loss is not None else 0.0

            total_loss = (
                    cls_loss
                    + self.args.alpha * orth_loss
                    + self.args.beta * concept_loss
            )

            total_loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total += batch_size
            correct += (cls_preds.argmax(dim=1) == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{100 * correct / total:.2f}%",
                'CLS': f"{cls_loss.item():.4f}",
                'CON': f"{concept_loss.item():.4f}",
                'ORT': f"{orth_loss.item():.4f}"
            })

        return {
            'train_acc': 100 * correct / total,
            'train_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'concept_loss': concept_loss.item(),
            'orth_loss': orth_loss.item()
        }

    def _evaluate(self, data_loader, mode='val'):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, attr_labels in tqdm(data_loader, desc=f"{mode.capitalize()} Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                attr_labels = attr_labels.to(self.device).float()

                concept_preds, cls_preds, reg_loss = self.model(images)

                cls_loss = self.loss_fns['cls'](cls_preds, labels)
                concept_loss = self.loss_fns['concept'](concept_preds, attr_labels)
                orth_loss = torch.abs(reg_loss).mean() if reg_loss is not None else 0.0
                total_loss = cls_loss + self.args.alpha * orth_loss + self.args.beta * concept_loss

                batch_size = images.size(0)
                total += batch_size
                correct += (cls_preds.argmax(dim=1) == labels).sum().item()

        return {
            f'{mode}_acc': 100 * correct / total,
            f'{mode}_loss': total_loss.item(),
            f'{mode}_cls_loss': cls_loss.item(),
            f'{mode}_concept_loss': concept_loss.item(),
            f'{mode}_orth_loss': orth_loss.item()
        }

    def run(self):
        logger.info("\n" + "=" * 60)
        logger.info(f"Starting Training for {self.args.epochs} epochs")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Model: {self.args.CBM_type}")
        logger.info(f"Alpha: {self.args.alpha}, Beta: {self.args.beta}")
        logger.info("=" * 60 + "\n")

        if self.args.wandb:
            wandb.init(
                project="LanCE",
                name=f"{self.args.dataset}-{self.args.CBM_type}-{time.strftime('%m%d%H%M')}",
                config=vars(self.args)
            )
            wandb.watch(self.model, log_freq=100)

        try:
            for epoch in range(self.args.epochs):
                epoch_start = time.time()

                train_metrics = self._train_epoch(epoch)

                source_metrics = self._evaluate(self.source_test_loader, 'source')
                target_metrics = self._evaluate(self.target_test_loader, 'target')

                if target_metrics['target_acc'] > self.best_metrics['target_acc']:
                    self.best_metrics.update({
                        'source_acc': source_metrics['source_acc'],
                        'target_acc': target_metrics['target_acc'],
                        'epoch': epoch
                    })
                    # self._save_concept_logs()
                    if self.args.save_model:
                        torch.save(self.model.state_dict(), f"best_{self.args.dataset}.pth")

                epoch_time = time.time() - epoch_start
                log_data = {
                    'epoch': epoch + 1,
                    'epoch_time': epoch_time,
                    **train_metrics,
                    **source_metrics,
                    **target_metrics
                }

                logger.info("\n" + "-" * 50)
                logger.info(f"Epoch {epoch + 1} Summary:")
                logger.info(f"Time: {epoch_time:.2f}s")
                logger.info(f"Train Acc: {train_metrics['train_acc']:.2f}%")
                logger.info(f"Source Acc: {source_metrics['source_acc']:.2f}%")
                logger.info(f"Target Acc: {target_metrics['target_acc']:.2f}%")
                logger.info(
                    f"Best Target Acc: {self.best_metrics['target_acc']:.2f}% @ Epoch {self.best_metrics['epoch'] + 1}")
                logger.info("-" * 50)

                if self.args.wandb:
                    wandb.log(log_data)

        except Exception as e:
            logger.error(f"Training interrupted: {str(e)}", exc_info=True)
            if self.args.wandb:
                wandb.alert(
                    title="Training Failed",
                    text=f"Error at epoch {epoch + 1}: {str(e)}"
                )
        finally:
            if self.args.wandb:
                wandb.summary.update(self.best_metrics)
                wandb.finish()

            logger.info("\n" + "=" * 60)
            logger.info("Training Completed")
            logger.info(f"Best Source Accuracy: {self.best_metrics['source_acc']:.2f}%")
            logger.info(f"Best Target Accuracy: {self.best_metrics['target_acc']:.2f}%")
            logger.info("=" * 60)


if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        session = TrainingSession(args)
        session.run()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Critical error occurred: {str(e)}", exc_info=True)