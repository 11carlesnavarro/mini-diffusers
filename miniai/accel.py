import torch
from miniai.learner import TrainCB, DeviceCB

class MixedPrecision(TrainCB):
    """
    A callback class for integrating mixed precision training into the training loop,
    which utilizes automatic mixed precision (AMP) from PyTorch for more efficient training on GPUs.
    
    Attributes:
        order (int): Specifies the execution order of this callback relative to others.
                     It is set to execute just after `DeviceCB` to ensure device adjustments are made prior.
    
    Methods:
        before_fit: Initializes the gradient scaler for AMP.
        before_batch: Enters the automatic mixed precision context before processing a batch.
        after_loss: Exits the automatic mixed precision context after calculating the loss.
        backward: Scales the loss before the backward pass to prevent underflow.
        step: Steps the optimizer and updates the scaler for the next batch.
    """
    order = DeviceCB.order + 1

    def before_fit(self, learn):
        """Initializes the gradient scaler for automatic mixed precision."""
        self.scaler = torch.cuda.amp.GradScaler()
    
    def before_batch(self, learn):
        """Enters the automatic mixed precision context."""
        self.autocast = torch.autocast("cuda", dtype=torch.float16)
        self.autocast.__enter__()
    
    def after_loss(self, learn):
        """Exits the automatic mixed precision context to proceed with the backward pass."""
        self.autocast.__exit__(None, None, None)

    def backward(self, learn):
        """Performs the backward pass with scaled loss to manage underflow."""
        self.scaler.scale(learn.loss).backward()
    
    def step(self, learn):
        """Steps the optimizer using the scaled gradients and updates the scaler for the next iteration."""
        self.scaler.step(learn.opt)
        self.scaler.update()


comment = """ class AccelerateCB(TrainCB):
    "
    A callback class for using the Accelerate library to simplify running PyTorch models on multi-GPUs,
    TPUs, or mixed precision training environments.
    
    Attributes:
        order (int): Execution order of this callback, set to execute after `DeviceCB`.
        n_inp (int): Number of inputs expected by the model, defaulting to 1.
        mixed_precision (str): The mixed precision mode to use, defaulting to "fp16".
        acc (Accelerator): An instance of the Accelerator class, configured for the mixed precision mode.
    
    Methods:
        before_fit: Prepares the model, optimizer, and dataloaders for distributed or mixed precision training.
        backward: Performs the backward pass using the Accelerator's optimized backward function.
    "
    order = DeviceCB.order + 1

    def __init__(self, n_inp=1, mixed_precision="fp16"):
        "Initializes the AccelerateCB with the number of inputs and mixed precision mode."
        super().__init__(n_inp=n_inp)
        self.acc = Accelerator(mixed_precision=mixed_precision)
    
    def before_fit(self, learn):
        "
        Prepares the model, optimizer, and dataloaders using the Accelerate library for optimized training.
        This includes adjustments for distributed training or mixed precision setups.
        "
        learn.model, learn.opt, learn.dls.train, learn.dls.valid = self.acc.prepare(
            learn.model, learn.opt, learn.dls.train, learn.dls.valid
        )

    def backward(self, learn):
        "Performs the backward pass using the Accelerate library's optimized function."
        self.acc.backward(learn.loss) """