"""
Evaluate a model on the training set.

Modified from evaluate.py.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import utilities.utils as U
from models import create_model
from utilities.configuration import CONSTANTS as C
from utilities.configuration import Configuration
from utilities.data import AMASSBatch, LMDBDataset
from utilities.data_transforms import ToTensor
from utilities.fk import SMPLForwardKinematics
from visualize import Visualizer


def load_model_weights(checkpoint_file, net, state_key="model_state_dict"):
    """Load a pre-trained model."""
    if not os.path.exists(checkpoint_file):
        raise ValueError("Could not find model checkpoint {}.".format(checkpoint_file))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    ckpt = checkpoint[state_key]
    net.load_state_dict(ckpt)


def get_model_config(model_id):
    """Get the model configuration of the specified model."""
    model_id = model_id
    model_dir = U.get_model_dir(C.EXPERIMENT_DIR, model_id)
    model_config = Configuration.from_json(os.path.join(model_dir, "config.json"))
    return model_config, model_dir


def load_model(model_id):
    """Load the specified model."""
    model_config, model_dir = get_model_config(model_id)
    net = create_model(model_config)

    net.to(C.DEVICE)
    print("Model created with {} trainable parameters".format(U.count_parameters(net)))

    # Load model weights.
    checkpoint_file = os.path.join(model_dir, "model.pth")
    load_model_weights(checkpoint_file, net)
    print("Loaded weights from {}".format(checkpoint_file))

    return net, model_config, model_dir


def evaluate_train(model_id, viz=False):
    """
    Load a model, evaluate it on the test set and save the predictions into the model directory.

    :param model_id: The ID of the model to load.
    :param viz: If some samples should be visualized.
    """
    net, model_config, model_dir = load_model(model_id)

    # No need to extract windows for the test set, since it only contains the seed sequence anyway.
    valid_transform = transforms.Compose([ToTensor()])
    valid_data = LMDBDataset(os.path.join(C.DATA_DIR, "validation"), transform=valid_transform)
    valid_loader = DataLoader(
        valid_data,
        batch_size=model_config.bs_eval,
        shuffle=False,
        num_workers=model_config.data_workers,
        collate_fn=AMASSBatch.from_sample_list,
    )

    # Put the model in evaluation mode.
    net.eval()
    results = dict()
    with torch.no_grad():
        for abatch in valid_loader:
            # Move data to GPU.
            if torch.cuda.is_available():
                batch_gpu = abatch.to_gpu()
            else:
                batch_gpu = abatch


            # Get the predictions.
            model_out = net(batch_gpu)

            for b in range(abatch.batch_size):
                results[batch_gpu.seq_ids[b]] = (
                    model_out["predictions"][b].detach().cpu().numpy(),
                    model_out["seed"][b].detach().cpu().numpy(),
                    abatch.poses[b].detach().cpu().numpy()[model_config.seed_seq_len :],
                )

    if viz:
        fk_engine = SMPLForwardKinematics()
        visualizer = Visualizer(fk_engine)
        n_samples_viz = 10
        rng = np.random.RandomState(42)
        idxs = rng.randint(0, len(results), size=n_samples_viz)
        sample_keys = [list(sorted(results.keys()))[i] for i in idxs]
        for k in sample_keys:
            visualizer.visualize_with_gt(
                results[k][1], results[k][0], results[k][2], title="Sample ID: {}".format(k)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="Which model to evaluate.")
    config = parser.parse_args()
    evaluate_train(config.model_id, viz=True)
