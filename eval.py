
import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from model_ import network_
from datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network_.GeoLocalizationNet_(args.backbone, args.fc_output_dim)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.info(f"Loading model_ from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the model_ (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model = model.to(args.device)

TEST_DATASETS = ["pitts250k", "pitts30k", "tokyo247", "msls", "st_lucia"]
for dataset_name in TEST_DATASETS:
    batchify = (dataset_name != 'tokyo247') # don't batchify only on tokyo
    try:
        all_ds_folder = args.test_set_folder.replace("sf_xl/processed/test", "")
        test_ds = TestDataset(f"{all_ds_folder}/{dataset_name}/images/test")
        logging.info(f"Testing set {dataset_name}: {test_ds}")

        recalls, recalls_str = test.test(args, test_ds, model, batchify)
        logging.info(f"{test_ds}: {recalls_str[:20]}")
    except Exception as e:
        logging.info(f"{dataset_name} with exception {e}")

test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold)

recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")
