import math

from epicon.optimizers import GradientDescent
from epicon.optimizers.schedulers import CosineAnnealingLR, StepLR


class TestStepLR:
    def test_lr_unchanged_before_step(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = StepLR(opt, step_size=10, gamma=0.1)
        scheduler.step(5)
        assert opt.current_learning_rate == 0.1

    def test_lr_decays_at_step(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = StepLR(opt, step_size=10, gamma=0.1)
        scheduler.step(10)
        assert abs(opt.current_learning_rate - 0.01) < 1e-15

    def test_lr_decays_multiple_steps(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = StepLR(opt, step_size=10, gamma=0.5)
        scheduler.step(20)
        assert opt.current_learning_rate == 0.025

    def test_lr_never_negative(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = StepLR(opt, step_size=5, gamma=0.1)
        scheduler.step(1000)
        assert opt.current_learning_rate >= 0


class TestCosineAnnealingLR:
    def test_lr_at_epoch_zero(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0)
        scheduler.step(0)
        assert opt.current_learning_rate == 0.1

    def test_lr_at_half_cycle(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0)
        scheduler.step(50)
        expected = 0.05 * (1 + math.cos(math.pi * 50 / 100))
        assert opt.current_learning_rate == expected

    def test_lr_at_full_cycle(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0)
        scheduler.step(100)
        assert opt.current_learning_rate == 0.0

    def test_lr_with_eta_min(self):
        opt = GradientDescent(learning_rate=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.01)
        scheduler.step(100)
        assert opt.current_learning_rate == 0.01
