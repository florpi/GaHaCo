"""
Utilities for reading in and amending config files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import os


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def load_config(config_file_path: str, purpose: str) -> dict:
    """
    Load and amend an experiment configuration.
    Args:
        config_file_path: Path to the JSON file containing the
            configuration to be loaded.
    Returns:
        A dictionary containing the amended configuration.
    """

    # -------------------------------------------------------------------------
    # Load configuration from JSON file
    # -------------------------------------------------------------------------

    # Build the full path to the config file and check if it exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"{config_file_path} does not exist!")

    # Load the config file into a dict
    with open(config_file_path, "r") as json_file:
        config = json.load(json_file)

    # -------------------------------------------------------------------------
    # Amend configuration (i.e., add implicitly defined variables)
    # -------------------------------------------------------------------------

    # Add the path to the experiments folder to the config dict
    if purpose is not 'optimize_tree':
        # Add the path to the experiments folder to the config dict
        config["experiment_dir"] = os.path.dirname(config_file_path)

    return config
