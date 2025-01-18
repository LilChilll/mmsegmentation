from mmcv.runner.hooks.hook import Hook,HOOKS
import os
import torch
from mmcv.fileio import FileClient
import os.path as osp
from mmcv.runner.dist_utils import allreduce_params,master_only
import warnings
import io
import mmcv
from mmcv.runner.checkpoint import get_state_dict,weights_to_cpu
from torch.optim import Optimizer
import time
from mmcv.parallel import is_module_wrapper
@HOOKS.register_module()
class CustomCheckpointHook(Hook):
    def __init__(self,
                 save_param_names,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 sync_buffer=False,
                 file_client_args=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args
        self.save_param_names = save_param_names

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

        runner.logger.info((f'Checkpoints will be saved to {self.out_dir} by '
                            f'{self.file_client.name}.'))

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        if 'create_symlink' in self.args:
            if self.args[
                    'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    ('create_symlink is set as True by the user but is changed'
                     'to be False because creating symbolic link is not '
                     f'allowed in {self.file_client.name}'))
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        # runner.save_checkpoint(
        #     self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        meta = {}
        meta.update(epoch=runner.epoch + 1, iter=runner.iter)
        filename_tmpl = 'iter_{}.pth'
        filename = filename_tmpl.format(runner.iter + 1)
        filepath = osp.join(self.out_dir, filename)
        optimizer = runner.optimizer if self.save_optimizer else None

        meta.update(mmcv_version=mmcv.__version__, time=time.asctime())
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if hasattr(model, 'CLASSES') and model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=model.CLASSES)
        model_state_dict = weights_to_cpu(get_state_dict(model))
        filtered_state_dict = {
            name: param for name, param in model_state_dict.items()
            if any(save_name in name for save_name in self.save_param_names)
        }
        checkpoint = {
            'meta': meta,
            'state_dict': filtered_state_dict
        }
        # save optimizer state dict in the checkpoint
        if isinstance(optimizer, Optimizer):
            checkpoint['optimizer'] = optimizer.state_dict()
        elif isinstance(optimizer, dict):
            checkpoint['optimizer'] = {}
            for name, optim in optimizer.items():
                checkpoint['optimizer'][name] = optim.state_dict()

        file_client_args = self.args.get('file_client_args', None)
        file_client = FileClient.infer_client(file_client_args, filepath)
        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            file_client.put(f.getvalue(), filepath)

        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)