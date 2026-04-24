from mmengine.registry import HOOKS

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.logging import MMLogger
from mmengine.model import MMDistributedDataParallel

from custom_mmdet.models.backbones.selective_vit import nnBlock
import os
from collections import OrderedDict
import torch
import torch.nn as nn



@HOOKS.register_module()
class LogBlockSparsity(Hook):
    def __init__(self, save_results: bool = False) -> None:
        self.logger: MMLogger = MMLogger.get_current_instance()
        self.save_results = save_results

    def save_results_in_file(self, runner, block_avg_values, block_dense_patches, block_sparse_patches, block_kr):
        try:
            result_file = os.path.join(runner.work_dir, 'block_sparsity_results.txt')
            with open(result_file, 'w') as f:
                if runner.train_loop is not None:
                    f.write("Meta Information Training:\n")
                    f.write(f"Epoch: {runner.train_loop.epoch}, Iteration: {runner.train_loop.iter}\n")
                    f.write(f"\n")

                f.write("Block Sparsity Metrics:\n")
                f.write(f"Average KR: {100 * block_avg_values['avg_kr']:.2f} %\n")
                f.write(f"Average Dense Patches: {block_avg_values['avg_dense_patches']}\n")
                f.write(f"Average Sparse Patches: {block_avg_values['avg_sparse_patches']}\n")

                f.write("\nPer-Block Metrics:\n")
                for name in block_dense_patches:
                    f.write(f"{name}:\n")
                    f.write(f"  KR: {100 * block_kr[name]:.2f} %\n")
                    f.write(f"  Dense Patches: {block_dense_patches[name]}\n")
                    f.write(f"  Sparse Patches: {block_sparse_patches[name]}\n")

        except Exception as e:
            self.logger.error(f"Failed to save block sparsity results: {e}")

    def get_model_from_runner(self, runner: Runner) -> str:
        if isinstance(runner.model,
                      (nn.DataParallel, torch.nn.parallel.DistributedDataParallel, MMDistributedDataParallel)):
            return runner.model.module
        return runner.model

    def before_val(self, runner:Runner) -> None:
        backbone = self.get_model_from_runner(runner)
        backbone = backbone.detector.backbone if hasattr(backbone, 'detector') else backbone
        for name, module in backbone.named_modules():
            if isinstance(module, nnBlock):
                module.enable_logging()

    def after_val(self, runner:Runner) -> None:
        backbone = self.get_model_from_runner(runner)
        backbone = backbone.detector.backbone if hasattr(backbone, 'detector') else backbone

        block_dense_patches = OrderedDict()
        block_sparse_patches = OrderedDict()
        block_kr = OrderedDict()

        for name, module in backbone.named_modules():
            if isinstance(module, nnBlock):
                num_dense_patches = module.metric_logger.meters["num_dense_patches"].avg
                num_selec_patches = module.metric_logger.meters["num_selec_patches"].avg
                block_dense_patches[name] = num_dense_patches
                block_sparse_patches[name] = num_selec_patches
                block_kr[name] = num_selec_patches / num_dense_patches

                module.disable_logging()
                y=1
        block_avg_values = OrderedDict()
        block_avg_values["avg_dense_patches"] = sum(block_dense_patches.values()) / len(block_dense_patches)
        block_avg_values["avg_sparse_patches"] = sum(block_sparse_patches.values()) / len(block_sparse_patches)
        block_avg_values["avg_kr"] = sum(block_kr.values()) / len(block_kr)

        self.logger.info("Block Sparsity Metrics: %.2f%% KR, %d dense patches, %d sparse patches"%
                         ((100 * block_avg_values["avg_kr"]), block_avg_values["avg_dense_patches"],
                                block_avg_values["avg_sparse_patches"]))
        blockwise_string = ", ".join([f"{key}: {100*value:.1f}" for key, value in block_kr.items()])
        self.logger.info("Block-wise KR: %s", blockwise_string)
        if self.save_results:
            self.save_results_in_file(runner, block_avg_values, block_dense_patches, block_sparse_patches, block_kr)

    def before_test(self, runner) -> None:
        self.before_val(runner)

    def after_test(self, runner) -> None:
        self.after_val(runner)

