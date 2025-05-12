import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf
from IPython.display import clear_output
from tensorflow import Tensor

from ..nca import NCA
from ..utils import pickle_save, raise_if
from ..utils.visualisation import display_image, save_image, to_rgb, horizontal_stack


class Trainer(ABC):
    """Abstract base class for trainer classes"""

    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        log_dir: str,
        seed: Optional[Tensor] = None,
    ):
        """
        Constructor for instances of `Trainer` instances

        Args:
            learning_rate (float): Learning rate to be used in training
            batch_size (int): Batch size used in training
            log_dir (str): Directory for training output to be saved in
            seed (Tensor, optional): Initial state of the NCA of shape (h,w,c).
                Note that `seed` must not include a batch dimension
        """
        raise_if(batch_size <= 0, "Batch size must be positive")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_dir = log_dir
        raise_if(
            not os.path.isdir(self.log_dir), f"Directory {self.log_dir} does not exist"
        )

        self.weights_dir = f"{self.log_dir}/model_weights"
        self.stages_dir = f"{self.log_dir}/stages"
        self.batches_dir = f"{self.log_dir}/batches"

        # Create directories if they do not exist
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.stages_dir, exist_ok=True)
        os.makedirs(self.batches_dir, exist_ok=True)

        self.seed = seed

        self.lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000], [self.learning_rate, self.learning_rate * 0.1]
        )
        self.optimizer = tf.keras.optimizers.Adam(self.lr_sched)  # Can be overridden in subclasses

        self.batch_visualisation_spacing = (
            0  # Spacing between the initial and final states in the batch visualisation
        )
        self.epoch: int

        self.losses = []
        self.save_points = [
            0,
            10,
            15,
            20,
            25,
            30,
            50,
            100,
            150,
            200,
            250,
        ]  # Default save points for generating stages

    @abstractmethod
    def train_loop(self, nca: NCA, epochs: int, start_epoch=0, **kwargs):
        """Training loop of trainer"""

    def train(self, nca: NCA, epochs: int = 8000, start_epoch=0, **kwargs):
        """
        Train the model

        Args:
            nca (NCA): Model to train
            epochs (int, optional): Number of epochs to train for. Defaults to 8000.
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
        """
        self.epoch = start_epoch
        try:
            self.train_loop(nca, epochs=epochs, start_epoch=start_epoch, **kwargs)
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {self.epoch}")

        self.save_model(nca, self.epoch)
        self.plot_loss(self.losses, save_path=f"{self.log_dir}/loss.png")
        nca.save_model_weights(f"{self.log_dir}/model_numpy_weights.pkl")
        state_dict = self.get_training_state()
        state_dict["model_name"] = nca.__class__.__name__
        pickle_save(state_dict, f"{self.log_dir}/training_state.pkl")

    def output(
        self,
        *,
        nca: NCA,
        epoch: int,
        loss: float,
        x0: Union[Tensor, ArrayLike],
        x: Union[Tensor, ArrayLike],
        major_n=100,
        minor_n=50,
        stages_save_points: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Helper function to produce output during training

        Args:
            nca (NCA): Model used to generate images
            epoch (int): Epoch number
            loss (float): Loss value
            x0 (Tensor | ArrayLike): Initial states of shape (b, h, w, c)
            x (Tensor | ArrayLike): Final states of same shape as `x0`
            major_n (int, optional):  How often to produce major output and save model weights. Defaults to 100.
            minor_n (int, optional): How often to product minor output. Defaults to 50.
            stages_save_points (List[int], optional): Points to generates when producing stages output. Defaults to None, in which case `Trainer.save_points` is used.
        """

        output_string = (
            f"\r Epoch: {epoch}. Loss: {loss}. Log Loss: {np.log10(loss):.3f}"
        )
        if kwargs:
            output_string += " " + ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
        if epoch % major_n == 0:
            clear_output()
            self.save_model(nca, epoch)
            self.plot_loss(self.losses)
            self.generate_stages(
                epoch,
                nca,
                seed=self.seed,
                stages_dir=self.stages_dir,
                save_points=stages_save_points,
            )
        if epoch % minor_n == 0:
            self.visualise_batch(
                x0,
                x,
                epoch,
                self.batches_dir,
                spacing=self.batch_visualisation_spacing,
                iterations=kwargs.get("iterations", None),
            )
        print(output_string, end="")

    def generate_stages(
        self,
        epoch: int,
        nca: NCA,
        seed: Optional[Tensor] = None,
        save_points: Optional[List[int]] = None,
        stages_dir: Optional[str] = None,
    ):
        """
        Generate image of the progression of NCA

        Args:
            epoch (int): Epoch number. This is used to generate file names.
            nca (NCA): Model uses to generate images
            seed (Tensor, optional): Initial starting state, of shape (h,w,c). If None, the current seed is used.
                Note that `seed` must not include a batch dimension
            save_points (List[int], optional): Points which to show the status of the NCA. Defaults to `Trainer.save_points`.
            log_dir (str, optional): Directory to save image in. Defaults to None, in which case `Trainer.stages_dir` is used.

        Raises:
            ValueError: If no seed is provided
        """
        if stages_dir is None:
            stages_dir = self.stages_dir
        seed = seed if seed is not None else self.seed
        if seed is None:
            raise ValueError("No seed provided")
        if save_points is None:
            save_points = self.save_points
        elif len(save_points) == 0:
            return  # No points to save
        frames = []
        x = seed[np.newaxis, ...]  # Introduce batch dimension to get (1,h,w,c)
        for i in range(max(save_points) + 1):
            x = nca(x)
            if i in save_points:
                frames.append(x[0])  # Remove batch dimension
        frames = np.array(frames)  # (b, h, w, c)
        vis = np.hstack(to_rgb(frames))  # Generate image
        display_image(vis)
        save_image(f"{stages_dir}/stages_{epoch:04d}.jpg", vis)

    def save_model(self, nca: NCA, epoch: int):
        """Save the model weights as a NumPy array

        Args:
            nca (NCA): Model to save
            epoch (int): Epoch number. This is used to generate the file name.
        """
        nca.save_weights(f"{self.weights_dir}/model_{epoch:04d}.weights.h5")

    def get_training_state(self) -> Dict[str, Any]:
        """Get the training state as a dictionary"""
        return dict(
            name=self.__class__.__name__,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epoch=self.epoch,
            losses=self.losses,
            stage_save_points=self.save_points,
            seed=self.seed,
        )

    @staticmethod
    def normalise_gradients(grads: List[Tensor]) -> List[Tensor]:
        """Normalise the gradients to help reduce instability in training

        Args:
            grads (List[Tensor]): List of gradients

        Returns:
            List[Tensor]: List of normalised gradients
        """
        return [g / (tf.norm(g) + 1e-8) for g in grads]

    @staticmethod
    def apply_gradients(
        nca: NCA, grads: List[Tensor], optimiser: tf.optimizers.Optimizer
    ):
        """
        Apply the computed gradients using the optimiser

        Args:
            nca (NCA): Model to update
            grads (List[Tensor]): Gradients of loss with respect to the weights
            optimiser (tf.optimizers.Optimizer): Optimiser to use to perform the update of the model weights
        """
        grads = Trainer.normalise_gradients(grads)
        optimiser.apply_gradients(zip(grads, nca.trainable_weights))

    @staticmethod
    def visualise_batch(
        x0: Union[Tensor, ArrayLike],
        x: Union[Tensor, ArrayLike],
        epoch: int,
        log_dir="train_log",
        spacing=0,
        iterations=None,
    ):
        """
        Visualise the batch progression

        Args:
            x0 (Tensor | ArrayLike): Initial states of shape (b, h, w, c) with c >= 4
            x (Tensor | ArrayLike): Final states of same shape as `x0`
            epoch (int): Epoch number. This is used to generate the file name
            log_dir (str, optional): Location to save the image. Defaults to 'train_log'.
        """
        vis0 = np.array(to_rgb(x0))
        vis1 = np.array(to_rgb(x))
        vis0 = horizontal_stack(vis0, spacing=spacing, pad_bottom=True)
        vis1 = horizontal_stack(vis1, spacing=spacing, pad_bottom=False)

        vis = np.vstack([vis0, vis1])  # Stack the initial and final states vertically
        if iterations:
            print(f"\nBatch before and after {iterations} iterations")
        else:
            print("\nBatch before and after")
        display_image(vis)
        save_image(f"{log_dir}/batches_{epoch:04d}.jpg", vis)

    @staticmethod
    def plot_loss(losses: List[float], save_path: Optional[str] = None):
        """Plot losses on graph in log space

        Args:
            losses (List[float]): List representing loss per epoch
            save_path (str, optional): Path to save the plot. If not provided, plot will not be saved.
        """
        plt.figure(figsize=(10, 4))
        plt.title("Loss history")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.plot(losses, ".", alpha=0.1)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
