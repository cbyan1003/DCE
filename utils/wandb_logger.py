import wandb
from wandb.wandb_run import Run
from argparse import Namespace
from collections import OrderedDict
import numpy as np
import torch.nn as nn

class WandbLogger:
    """
    Wandb logger class to monitor training.

    Parameters
    ----------
    name : str
        Run name (if empty, uses a fancy Wandb name, highly recommended)
    dir : str
        Folder where wandb information is stored
    id : str
        ID for the run
    anonymous : bool
        Anonymous mode
    version : str
        Run version
    project : str
        Wandb project where the run will live
    tags : list of str
        List of tags to append to the run
    log_model : bool
        Log the model to wandb or not
    experiment : wandb
        Wandb experiment
    entity : str
        Wandb entity
    """
    def __init__(self, cfg,
                 name=None, dir=None, id=None, anonymous=False,
                 project=None, entity=None,
                 tags=None, log_model=False, experiment=None
                 ):
        super().__init__()
        self._name = name
        self._dir = dir
        self._anonymous = 'allow' if anonymous else 'never'
        self._id = id
        self._tags = tags
        self._project = project
        self._entity = entity
        self._log_model = log_model

        self._experiment = experiment if experiment else self.create_experiment(cfg)
        self.config = wandb.config
        self._metrics = OrderedDict()
    
    def __getstate__(self):
        """Get the current logger state"""
        state = self.__dict__.copy()
        state['_id'] = self._experiment.id if self._experiment is not None else None
        state['_experiment'] = None
        return state

    def create_experiment(self, cfg):
        """Creates and returns a new experiment"""
        experiment = wandb.init(config=cfg,
            name=self._name, dir=self._dir, project=self._project,
            anonymous=self._anonymous, reinit=True, id=self._id,
            resume='allow', tags=self._tags, entity=self._entity,
            group='DDP_4GPU_test'
        )
        # wandb.save(self._dir)
        return experiment

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100, log_graph:bool = False):
        """Watch training parameters."""
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    @property
    def experiment(self) -> Run:
        """Returns the experiment (creates a new if it doesn't exist)."""
        if self._experiment is None:
            self._experiment = self.create_experiment()
        return self._experiment

    @property
    def version(self) -> str:
        """Returns experiment version."""
        return self._experiment.id if self._experiment else None

    @property
    def name(self) -> str:
        """Returns experiment name."""
        name = self._experiment.project_name() if self._experiment else None
        return name

    @property
    def run_name(self) -> str:
        """Returns run name."""
        return wandb.run.name if self._experiment else None

    @property
    def finish(self):
        return wandb.finish()
    
    @property
    def run_url(self) -> str:
        """Returns run URL."""
        return 'https://app.wandb.ai/{}/{}/runs/{}'.format(
            wandb.run.entity, wandb.run.project, wandb.run.id) if self._experiment else None
        
    def add_scalar(self, name, value, global_step):
        self.log_metrics({name:value, 'global_step':global_step})
        
    def add_figure(self,name,image, global_step):
        self.log_metrics({name: wandb.Image(image), 'global_step':global_step})
        
    def log_metrics(self, metrics):
        """Logs training metrics."""
        self._metrics.update(metrics)
        if 'global_step' in metrics:
            self.experiment.log(self._metrics)
            self._metrics.clear()
            
    