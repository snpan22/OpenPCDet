import torch
import torch.nn.functional as F
import argparse
import pickle as pkl
import logging
import json
from pathlib import Path
import traceback
import os
import sys
from tqdm import tqdm
from pcdet.utils import common_utils
import numpy as np

import importlib
import helpers_ptt
importlib.reload(helpers_ptt)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logger(log_path):

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_file', default="waymo_eval.log")
    parser.add_argument('--metrics_out', default="metrics.json")
    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_evaluations(args, logger):

    # args = parse_args()
    # logger = setup_logger(args.log_file)

    logger.info("Loading config...")
    cfg_from_yaml_file(args.cfg_file, cfg)

    logger.info("Building dataset (GT only, no model)...")
    
    
    CFG_FILE = args.cfg_file
    CKPT = args.ckpt

    cfg_from_yaml_file(CFG_FILE, cfg)
    cfg.TAG = Path(CFG_FILE).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'

    logger.info(f'Loaded cfg from {CFG_FILE}')


    dataset, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False
    )

    len_test = len(dataset)
    logger.info(f'Test set length: {len_test}')
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )

    logger.info(f'Loading checkpoint from: {CKPT}')
    model.load_params_from_file(filename=CKPT, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    
    
    global_to_segment = {}
    segment_frame_counters = {}
    for i, info in enumerate(dataset.infos):
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        global_to_segment[i] = (seg, segment_frame_counters[seg])
        segment_frame_counters[seg] += 1
    
    result_file = "output/cfgs/custom_models/centerpoint_singleframe_waymo/default/eval/eval_with_train/epoch_30/val/result.pkl"
    with open(result_file, "rb") as f:
        det_annos = pkl.load(f)
    # --------------------------------------------------------
    # Progress indicator during evaluation
    # --------------------------------------------------------
    # dataset.evaluation() is monolithic, so we simulate progress
    # by wrapping the call — still useful for tracking runtime
    # --------------------------------------------------------

    logger.info("\n\nStarting Waymo evaluation")

    for _ in tqdm(range(1), desc="Waymo Metrics"):
        result_str, result_dict = dataset.evaluation(
            det_annos,
            cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

    logger.info("Evaluation complete")

    logger.info("\n===== RESULT STRING =====\n")
    logger.info("\n" + result_str)

    logger.info("\n===== RESULT DICT =====")
    logger.info(json.dumps(result_dict, indent=2, default = float))

    # Save metrics JSON
    with open(args.metrics_out, "w") as f:
        json.dump(result_dict, f, indent=2, default = float)

    logger.info(f"Metrics saved to {args.metrics_out}")

def main():

    args = parse_args()
    logger = setup_logger(args.log_file)

    try:
        run_evaluations(args, logger)

    except Exception as e:
        logger.error("===== EVALUATION FAILED =====")
        logger.error(str(e))
        logger.error("\nFull traceback:\n")
        logger.error(traceback.format_exc())

        # Also print to stderr so Slurm captures it
        print(traceback.format_exc(), file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()