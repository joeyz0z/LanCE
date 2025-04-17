import os
import time
import json
import logging
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb

os.environ["WANDB_MODE"] = "offline"

# Local imports
from args import get_args
import clip
from model.cbm_models import clipzs
from data import get_dataset_classes
from prompts.prompt200new import source_text_prompts, target_text_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("zeroshot_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClipZeroShotTester:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.args.device = self.device
        self._init_components()

        logger.info("\n" + "=" * 60)
        logger.info(f"CLIP Zero-Shot Test Configuration")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Prompt Type: {args.prompt_type}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info("=" * 60 + "\n")

    def _setup_device(self):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")
        return device

    def _init_components(self):
        self._prepare_datasets()
        self._init_model()

    def _prepare_datasets(self):
        """Load dataset based on configuration"""
        self.train_dataset, self.train_loader, self.source_test_dataset, self.source_test_loader, self.target_test_dataset, self.target_test_loader = get_dataset_classes(self.args)

        logger.info(f"Dataset {self.args.dataset} loaded")
        logger.info(f"Source test samples: {len(self.source_test_loader.dataset):,}")
        logger.info(f"Target test samples: {len(self.target_test_loader.dataset):,}")

    def _init_model(self):
        """Initialize CLIP zero-shot model with domain prompts"""
        prompt_texts = source_text_prompts if self.args.prompt_type == "origin" else target_text_prompts

        self.model = clipzs(
            self.args,
            prompt_texts,
            list(self.train_dataset.classname2id.keys()),
            list(self.train_dataset.concept2id.keys()),
            self.train_dataset.domain_diffs
        ).to(self.device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        logger.info(f"Model initialized with {self.args.prompt_type} prompts")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _evaluate_accuracy(self, data_loader, mode='source'):
        """Evaluate model accuracy on given data loader"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in tqdm(data_loader, desc=f"{mode.capitalize()} Evaluation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images)
                correct += (preds.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        return round(accuracy, 2)

    def run_test(self):
        """Run zero-shot evaluation on both domains"""
        try:
            source_acc = self._evaluate_accuracy(self.source_test_loader, 'source')
            target_acc = self._evaluate_accuracy(self.target_test_loader, 'target')

            logger.info("\n" + "=" * 60)
            logger.info(f"Zero-Shot Test Results:")
            logger.info(f"Source Domain Accuracy: {source_acc}%")
            logger.info(f"Target Domain Accuracy: {target_acc}%")
            logger.info("=" * 60)

            if self.args.wandb:
                wandb.log({
                    "source_acc": source_acc,
                    "target_acc": target_acc
                })

            return source_acc, target_acc

        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=True)
            if self.args.wandb:
                wandb.alert(
                    title="Zero-Shot Test Failed",
                    text=str(e)
                )
            raise


if __name__ == "__main__":
    args = get_args()

    # Seed configuration
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if args.wandb:
            wandb.init(
                project="CLIP-ZeroShot",
                name=f"{args.dataset}-{args.prompt_type}-{time.strftime('%m%d%H%M')}",
                config=vars(args)
            )

        tester = ClipZeroShotTester(args)
        tester.run_test()

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Critical error occurred: {str(e)}", exc_info=True)
    finally:
        if args.wandb:
            wandb.finish()