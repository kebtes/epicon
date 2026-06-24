import math


class StepLR:
    """
    Decays the learning rate by a factor every step_size epochs.

    Parameters:
        optimizer (Optimizer): The optimizer to schedule.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
                       Defaults to 0.1.

    Examples:
        >>> from epicon.optimizers import GradientDescent, schedulers
        >>> opt = GradientDescent(learning_rate=0.1)
        >>> scheduler = schedulers.StepLR(opt, step_size=30, gamma=0.1)
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self._base_lr = optimizer.learning_rate

    def step(self, epoch: int):
        """
        Update the optimizer's learning rate based on the current epoch.

        Parameters:
            epoch (int): Current epoch number (1-indexed).
        """
        factor = self.gamma ** (epoch // self.step_size)
        self.optimizer.current_learning_rate = self._base_lr * factor


class CosineAnnealingLR:
    """
    Sets the learning rate using a cosine annealing schedule.

    Parameters:
        optimizer (Optimizer): The optimizer to schedule.
        T_max (int): Maximum number of epochs (half-cycle period).
        eta_min (float): Minimum learning rate. Defaults to 0.

    Examples:
        >>> from epicon.optimizers import Adam, schedulers
        >>> opt = Adam(learning_rate=0.001)
        >>> scheduler = schedulers.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self._base_lr = optimizer.learning_rate

    def step(self, epoch: int):
        """
        Update the optimizer's learning rate based on the current epoch.

        Parameters:
            epoch (int): Current epoch number (1-indexed).
        """
        cos_val = math.cos(math.pi * epoch / self.T_max)
        self.optimizer.current_learning_rate = self.eta_min + (self._base_lr - self.eta_min) * (1 + cos_val) / 2
