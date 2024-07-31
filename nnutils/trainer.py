"""
A simple trainer class method that allows for easily transferring major
utilities/boiler plate code.
"""
import os
import random
import time

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
# import wandb
from tqdm import tqdm

from datasets.builder import build_loader
from models.build_model import build_model
from utils.io import makedir
from datasets.builder import get_sample_dataloader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from utils.wandb_logger import WandbLogger
import wandb

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "22345"

def none_grad(model):
    for p in model.parameters():
        p.grad = None


class BaseEngine:
    """
    Basic engine class that can be extended to be a trainer or evaluater.
    Captures default settings for building losses/metrics/optimizers.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # assuming single gpu jobs for now
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = build_model(cfg).to(self.device)
        self.model = build_model(cfg)
    

class BasicTrainer(BaseEngine):
    def __init__(self, cfg):
        super(BasicTrainer, self).__init__(cfg)
        # For reproducibility -
        # refer https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(cfg.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)

        # Define dataset loaders
        # Overfitting steps instead of epochs
        # if cfg.overfit:
        #     self.train_loader = build_loader(cfg, split="train", overfit=100)
        #     self.valid_loader = build_loader(cfg, split="train", overfit=1)
        # else:
        #     self.train_loader = build_loader(cfg, split="train")
        #     self.valid_loader = build_loader(cfg, split="valid")

        # get a single instance; just for debugging purposes
        # self.train_loader.dataset.__getitem__(0)
        # self.valid_loader.dataset.__getitem__(0)

        # Define some useful training parameters
        self.epoch = 0
        self.step = 0
        self.eval_step = 100 if cfg.overfit else cfg.eval_step
        self.num_epochs = cfg.num_epochs
        self.vis_step = cfg.vis_step
        self.best_loss = 1e9  # very large number
        self.curr_loss = 1e8  # slightly smaller very large number
        self.training = True

        # Define Solvers
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Restore from checkpoint if a path is provided
        if cfg.checkpoint != "":
            self.restore_checkpoint(cfg.checkpoint)
            self.full_exp_name = "resumed_{}_{:}".format(
                cfg.EXPname, time.strftime("%m%d-%H%M"),
            )
        else:
            if cfg.EXPname == "DEBUG":
                self.full_exp_name = cfg.EXPname
            else:
                self.full_exp_name = "{}_{:}".format(
                    cfg.EXPname, time.strftime("%m%d-%H%M"),
                )
            print("Full experiment name: {}".format(self.full_exp_name))

        
        # Define logger
        # self.logger = None
        # self.logger = wandb.init(config = self.cfg,
        #                          name = self.full_exp_name,
        #                          dir = self.cfg.wandb_dir,
        #                          id = self.full_exp_name,
        #                          anonymous = None,
        #                          project = self.cfg.wandb_project,
        #                          entity = self.cfg.wandb_entity,
        #                          tags = self.cfg.wandb_tags,
        #                          resume = 'allow',
        #                          reinit = True)


        # 
        # self.logger = SummaryWriter(log_dir=log_dir)

        # Define experimental dir for checkpoints
        exp_dir = os.path.join(cfg.experiments_dir, self.full_exp_name)
        makedir(exp_dir, replace_existing=True)
        self.experiment_dir = exp_dir
        
    
    def get_logger(self, cfg):
        if cfg.logging_method == 'wandb':
            if cfg.wandb_dry_run is True:
                print('use logger: none (dry_run is true)')
                return None
            print('use logger:', cfg.logging_method)
            
            log_dir = cfg.wandb_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            logger = WandbLogger(cfg,
                                 name = self.full_exp_name,
                                 dir = log_dir,
                                 id = self.full_exp_name,
                                 anonymous = cfg.wandb_dry_run,
                                 project = cfg.wandb_project,
                                 entity = cfg.wandb_entity,
                                 tags = cfg.wandb_tags,
                                 log_model = False,
                                 experiment = None)
        return logger
    
    def save_checkpoint(self):
        if self.step == 0:
            return
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()

        # checkpoint_dict
        checkpoint = {
            "model": model_state,
            "optim": optim_state,
            "curr_loss": self.curr_loss,
            "best_loss": self.best_loss,
            "epoch": self.epoch,
            "step": self.step,
            "cfg": self.cfg,
        }

        name = "checkpoint@e{:04d}s{:07d}.pkl".format(self.epoch, self.step)
        path = os.path.join(self.experiment_dir, name)

        print("Saved a checkpoint {}".format(name))
        torch.save(checkpoint, path)

        if self.curr_loss == self.best_loss:
            # Not clear if best loss is best accuracy, but whatever
            print("Best model so far, with a loss of {}".format(self.best_loss))
            path = os.path.join(self.experiment_dir, "best_loss.pkl")
            torch.save(checkpoint, path)

        # return model to state
        self.model.to(self.device)

    def restore_checkpoint(self, checkpoint_path):
        print("Restoring checkpoint {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.curr_loss = checkpoint["curr_loss"]
        self.best_loss = checkpoint["best_loss"]

        # update model params
        old_dict = checkpoint["model"]
        model_dict = {}
        for k in old_dict:
            if "module" == k[0:6]:
                model_dict[k[7:]] = old_dict[k]
            else:
                model_dict[k] = old_dict[k]

        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        # update optim params
        self.optimizer.load_state_dict(checkpoint["optim"])

    def build_optimizer(self, network=None):
        # Currently just taking all the model parameters
        cfg = self.cfg
        if network is None:
            network = self.model
        params = network.parameters()

        # Define optimizer
        if cfg.optimizer == "SGD":
            return torch.optim.SGD(
                params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=False,
            )
        elif cfg.optimizer == "Adam":
            return torch.optim.Adam(
                params, lr=cfg.lr, eps=1e-4, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "RMS":
            return torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            raise ValueError()

    def build_scheduler(self):
        cfg = self.cfg

        if cfg.scheduler == "constant":
            # setting gamma to 1 means constant LR
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=1
            )
        else:
            raise ValueError("Scheduler {} not recognized".format(cfg.scheduler))

        return scheduler

    def _prepare_metric(self, x):
        x = x.detach().cpu()
        return x

    def _update_metrics(self, epoch_metrics, batch_metrics):
        # Update metrics
        for k in batch_metrics:
            b_metrics = self._prepare_metric(batch_metrics[k])

            if k in epoch_metrics:
                epoch_metrics[k] = torch.cat((epoch_metrics[k], b_metrics), dim=0,)
            else:
                epoch_metrics[k] = b_metrics
        return epoch_metrics

    def log_dict(self, log_dict, header, split, logger):
        if logger is None:
            return
        for key in log_dict:
            val = log_dict[key]
            if "torch" in str(type(val)):
                val = val.mean().item()

            if split is None:
                tab = header
            else:
                key_parts = key.split("_")
                if np.isscalar(key_parts[-1]):
                    tab = "_".join(key_parts[:-1])
                else:
                    tab = key
                key = key + "_" + split
            if np.isscalar(val) or len(val.shape) == 1:
                logger.add_scalar("{}/{}".format(tab, key), val, self.step)
            elif len(val.shape) in [2, 3]:
                pass
            else:
                raise ValueError("Cannot log {} on tensorboard".format(val))

    def forward_preprocess(self):
        pass

    def calculate_norm_dict(self):
        grad_fn = torch.nn.utils.clip_grad_norm_
        grad_norm = grad_fn(self.model.parameters(), 10 ** 20)
        return {"full_model": grad_norm.item()}

    def setup(self, rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    def cleanup(self):
        dist.destroy_process_group()
    
    def train_epoch(self, rank, world_size, DDP_model, train_sample_loader, logger=None):
        epoch_loss = 0
        e_metrics = {}
        norm_dict = {}
        time_dict = {}
        d_metrics = {}
        DDP_model.train()
        
        # Setup tqdm loader -- no description for now
        disable_tqdm = rank != 0 
        t_loader = tqdm(train_sample_loader, disable=disable_tqdm, dynamic_ncols=True,)

        # Begin training
        before_load = time.time()
        for i, batch in enumerate(t_loader):
            self.forward_preprocess()
            none_grad(DDP_model)
            after_load = time.time()
            self.step += 1
            # print(f"##### data load NEED: {after_load - before_load} seconds#####")
            for k, v in batch.items():
                if type(v) == list:
                    if k == 'path_0' or k == 'path_1' or k == 'neighbor_len' or k == 'p2i_list_len':
                        continue
                    batch[k] = [item.to(rank) for item in v]
                else:
                    batch[k] = v.to(rank)
                
            # Forward pass
            # try:
            #     b_loss, b_metrics, b_outputs = self.forward_batch(batch)
            # except Exception as e:
            #     print('error:', e)
            #     continue
            
            b_loss, b_metrics, b_outputs = self.forward_batch(batch)
            
            after_forward = time.time()
            
            # Backward pass
            b_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            after_backward = time.time()
            norm_dict = self.calculate_norm_dict()

            # calculate times
            load_time = after_load - before_load
            fore_time = after_forward - after_load
            back_time = after_backward - after_forward
            total_time = load_time + fore_time + back_time

            time_dict["total_time"] = total_time
            time_dict["data_ratio"] = load_time / total_time
            time_dict["fore_ratio"] = fore_time / total_time
            time_dict["back_ratio"] = back_time / total_time

            # Log results
            e_metrics = self._update_metrics(e_metrics, b_metrics)
            e_metrics = self._update_metrics(e_metrics, d_metrics)
            epoch_loss += b_loss.item()
            if logger != None:
                self.log_dict(b_metrics, "metrics", "train", logger)
                self.log_dict(d_metrics, "metrics", "train", logger)
                self.log_dict(norm_dict, "grad_norm", None, logger)
                self.log_dict(time_dict, "time", None, logger)
                logger.add_scalar("train/epoch", self.epoch, self.step)
                current_lr = self.scheduler.get_last_lr()[0]
                logger.add_scalar("train/lr", current_lr, self.step)

            # # Validate and save checkpoint based on step?
            # if (self.step % self.eval_step) == 0:
            #     none_grad(self.model)
            #     self.validate()
            #     self.model.train()

            # reset timer
            before_load = time.time()

        # Log results
        # Scale down calculate metrics
        epoch_loss /= len(train_sample_loader)
        print("Training Metrics: ")
        metric_keys = list(e_metrics.keys())
        metric_keys.sort()
        for m in metric_keys:
            print("    {:25}:   {:10.5f}".format(m, e_metrics[m].mean().item()))
    
    def test_epoch(self, rank, world_size, train_sample_loader, logger=None):
        # Setup tqdm loader -- no description for now
        disable_tqdm = rank != 0 
        t_loader = tqdm(train_sample_loader, disable=disable_tqdm, dynamic_ncols=True,)

        for i, batch in enumerate(t_loader):
            self.forward_preprocess()
            self.step += 1
            print(self.step)


    def validate(self, rank, world_size, DDP_model, valid_sample_loader, split="valid", logger=None):
        v_loss = torch.zeros(1).to(self.device)
        v_metrics = {}
        self.training = False
        DDP_model.eval()

        # Setup tqdm loader -- no description for now
        disable_tqdm = not self.cfg.TQDM
        tqdm_loader = tqdm(valid_sample_loader, disable=disable_tqdm, dynamic_ncols=True,)

        for i, batch in enumerate(tqdm_loader):
            for k, v in batch.items():
                if type(v) == list:
                    batch[k] = [item.to(rank) for item in v]
                else:
                    batch[k] = v.to(rank)
            
            # Forward pass
            b_loss, b_metrics, b_outputs = self.forward_batch(batch)

            # Log results
            v_metrics = self._update_metrics(v_metrics, b_metrics)
            v_loss += b_loss.detach()
  

        # Scale down calculate metrics
        v_loss /= len(valid_sample_loader)

        for k in v_metrics:
            # very hacky -- move to gpu to sync; mean first to limit gpu memory
            v_metrics[k] = v_metrics[k].mean().to(rank)

        # Log results
        if logger != None:
            self.log_dict(v_metrics, "metrics", split, logger)
            print("Validation after {} epochs".format(self.epoch))
            print("  {} Metrics: ".format(split))
            metric_keys = list(v_metrics.keys())
            metric_keys.sort()
            for m in metric_keys:
                print("    {:25}:   {:10.5f}".format(m, v_metrics[m].mean().item()))

        self.curr_loss = v_loss
        # Update best loss
        if v_loss < self.best_loss:
            self.best_loss = v_loss

        # save model
        self.save_checkpoint()

        # Restore training setup
        self.training = True
        return v_metrics
    def ddp_train(self):
        world_size = torch.cuda.device_count()
        mp.spawn(self.train, args={world_size,}, nprocs=world_size, join=True)
        
    def train(self, rank, world_size):
        # self.validate()
        test = False
        self.setup(rank, world_size)

        # Only rank 0 should log
        if rank ==0:
            logger = self.get_logger(self.cfg)
            
        self.device = torch.device(f'cuda:{rank}')
        
        model = self.model.to(self.device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
        if not test:
            DDP_model = DDP(model, device_ids=[rank])

        train_sample_loader, sample_neighborhood = get_sample_dataloader(self.cfg, split="train", rank=rank, world_size=world_size)
        valid_sample_loader, _ = get_sample_dataloader(self.cfg, split="valid", neighborhood_limits=sample_neighborhood, rank=rank, world_size=world_size)
        
        for _ in range(self.epoch, self.num_epochs):
            train_sample_loader.sampler.set_epoch(self.epoch)
            if test:
                self.test_epoch(rank, world_size, train_sample_loader)
            else:
                if rank ==0:
                    self.train_epoch(rank, world_size, DDP_model, train_sample_loader, logger)
                else:
                    self.train_epoch(rank, world_size, DDP_model, train_sample_loader)
                self.epoch += 1
                if rank ==0:
                    self.validate(rank, world_size, DDP_model, valid_sample_loader, logger)
                else:
                    self.validate(rank, world_size, DDP_model, valid_sample_loader)
        self.cleanup()
        if rank ==0:
            logger.finish()
            
