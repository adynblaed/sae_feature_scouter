#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_api_services.py:
End-to-End Test Script for SAE Visualizer Toolkit API.

This script tests the complete API functionality by:
  - Generating dashboards for various demo types:
      [1] Basic Feature-Centric Visualization
      [2] Custom Layout Visualization
      [3] Prompt-Centric Visualization
      [4] OthelloGPT Visualization (with linear probes)
      [5] Attention-Only Model Visualization
  - Preparing model-specific data as needed.
  - Removing pre-existing output files if requested.
  - Logging detailed information and timing each test.
  - Producing a JSON report summarizing the test outcomes.

The output HTML files are saved to a directory specified by --output-dir
(defaulting to "dashboards").

This design is robust, idempotent, and CI/CD–friendly.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import torch
from datasets import load_dataset
from sae_feature_scouter.services.workflow import run_workflow
from sae_feature_scouter.services.model_loader import load_othello_model, load_attn_model
from huggingface_hub import hf_hub_download

# Configure logging for the tests.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_api_services")


def prepare_othello_data(device: str):
    """
    Prepares OthelloGPT-related data by:
      - Downloading tokens, target_logits, and linear_probes files.
      - Running a forward pass to compute 'alive' features.
    
    Returns:
        tuple: (othello_tokens, othello_target_logits, othello_linear_probes, othello_alive_feats)
    """
    hf_repo_id = "callummcdougall/arena-demos-othellogpt"
    sae_id = "blocks.5.mlp.hook_post-v1"

    def hf_othello_load(filename: str):
        path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
        return torch.load(path, weights_only=True, map_location=device)

    othello_tokens = hf_othello_load("tokens.pt")[:5000]
    othello_target_logits = hf_othello_load("target_logits.pt")[:5000]
    othello_linear_probes = hf_othello_load("linear_probes.pt")
    
    # Load OthelloGPT model and its SAE.
    data = load_othello_model()
    model = data["model"]
    sae = data["sae"]
    
    hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    _, cache = model.run_with_cache_with_saes(
        othello_tokens[:128],
        saes=[sae],
        names_filter=hook_name,
    )
    acts = cache[hook_name]
    othello_alive_feats = (acts[:, 5:-5].flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()
    
    del cache
    torch.cuda.empty_cache()
    gc.collect()
    return othello_tokens, othello_target_logits, othello_linear_probes, othello_alive_feats


def prepare_attn_data(device: str):
    """
    Prepares data for the attention-only model by:
      - Loading a streaming dataset.
      - Tokenizing and prepending the BOS token.
      - Running inference to compute 'alive' features.
    
    Returns:
        tuple: (tokens, attn_alive_feats)
    """
    data = load_attn_model()  # Returns a dict with keys "model" and "sae"
    model = data["model"]
    sae = data["sae"]

    original_dataset = load_dataset(sae.cfg.dataset_path, split="train", streaming=True, trust_remote_code=True)
    batch_size = 256
    seq_len = 64
    seq_list = [x["input_ids"][: seq_len - 1] for (_, x) in zip(range(batch_size), original_dataset)]
    tokens = torch.tensor(seq_list, device=device)
    bos_token = torch.tensor([model.tokenizer.bos_token_id for _ in range(batch_size)], device=device)
    tokens = torch.cat([bos_token.unsqueeze(1), tokens], dim=1)
    
    hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    _, cache = model.run_with_cache_with_saes(
        tokens[:64],
        saes=[sae],
        names_filter=hook_name,
        stop_at_layer=sae.cfg.hook_layer + 1,
    )
    acts = cache[hook_name]
    attn_alive_feats = (acts.flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()

    del cache
    torch.cuda.empty_cache()
    gc.collect()
    return tokens, attn_alive_feats


def remove_output_file(filepath: str):
    """
    Remove an output file if it exists to ensure idempotence.
    
    Args:
        filepath (str): Full path of the file to remove.
    """
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info("Removed existing output file: %s", filepath)
        except Exception as e:
            logger.error("Failed to remove output file '%s': %s", filepath, str(e))


def run_test(test_num: int, device: str, output_dir: str) -> dict:
    """
    Runs an individual test based on its number and returns a result dictionary.
    
    Args:
        test_num (int): The demo type/test number (1 to 5).
        device (str): Computation device ("cuda" or "cpu").
        output_dir (str): Directory to save dashboard HTML files.
    
    Returns:
        dict: Contains keys 'test', 'status', 'message', 'output', and 'time_taken'.
    """
    result = {"test": test_num, "status": "fail", "message": "", "output": None, "time_taken": 0}
    start_time = time.time()

    # Map demo types to output filenames.
    filenames = {
        1: "demo_feature_vis.html",
        2: "demo_feature_vis_custom.html",
        3: "demo_prompt_vis.html",
        4: "demo_othello_vis.html",
        5: "demo_feature_vis_attn2l.html"
    }
    if test_num not in filenames:
        result["message"] = f"Test {test_num} is not defined."
        return result

    output_file = os.path.join(output_dir, filenames[test_num])
    if args.clean:
        remove_output_file(output_file)

    try:
        if test_num == 1:
            out = run_workflow(1, output_filename=output_file)
            result["output"] = out
        elif test_num == 2:
            out = run_workflow(2, output_filename=output_file)
            result["output"] = out
        elif test_num == 3:
            out = run_workflow(3, output_filename=output_file, prompt="'first_name': ('django.db.models.fields", metric="act_quantile")
            result["output"] = out
        elif test_num == 4:
            othello_tokens, _, _, _ = prepare_othello_data(device)
            out = run_workflow(4, output_filename=output_file, feature=8, othello_tokens=othello_tokens)
            result["output"] = out
        elif test_num == 5:
            tokens, _ = prepare_attn_data(device)
            out = run_workflow(5, output_filename=output_file, feature=8, tokens=tokens)
            result["output"] = out
        result["status"] = "pass"
    except Exception as e:
        result["message"] = str(e)
        logger.error("Test %d failed: %s", test_num, str(e), exc_info=True)
    result["time_taken"] = time.time() - start_time
    return result


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the test script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="End-to-End Test Script for SAE Visualizer Toolkit API."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (cuda or cpu)")
    parser.add_argument("--tests", type=str, default="1,2,3,4,5",
                        help="Comma-separated list of test numbers to run (default: 1,2,3,4,5)")
    parser.add_argument("--output-dir", type=str, default="dashboards",
                        help="Directory to output dashboard HTML files (default: dashboards)")
    parser.add_argument("--report-file", type=str, default="",
                        help="Optional path to output a JSON report file")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing output files before running tests")
    return parser.parse_args()


def main():
    global args  # Make args available globally for run_test
    args = parse_args()
    device = args.device

    # Ensure output directory exists.
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            logger.info("Created output directory: %s", args.output_dir)
        except Exception as e:
            logger.error("Failed to create output directory '%s': %s", args.output_dir, str(e))
            sys.exit(1)

    test_numbers = [int(t.strip()) for t in args.tests.split(",") if t.strip().isdigit()]
    overall_results = []
    logger.info("Starting End-to-End API tests on device: %s", device)
    
    for test_num in test_numbers:
        logger.info("Running Test %d...", test_num)
        result = run_test(test_num, device, args.output_dir)
        overall_results.append(result)
        logger.info("Test %d completed with status: %s in %.2fs", test_num, result["status"], result["time_taken"])
    
    # Build summary report.
    passed = sum(1 for r in overall_results if r["status"] == "pass")
    total = len(overall_results)
    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": overall_results,
        "timestamp": time.time()
    }
    logger.info("Test Summary: %d/%d tests passed.", passed, total)
    print(json.dumps(summary, indent=2))
    
    if args.report_file:
        try:
            with open(args.report_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Report saved to %s", args.report_file)
        except Exception as e:
            logger.error("Failed to write report file: %s", str(e))
    
    # Exit with non-zero code if any test failed.
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
