import yaml
import os
import argparse


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_config(expt_config_name: str):
    """ takes .yaml file name for the experiment config, 
    modifies the main config with it and returns a combined configureation"""

    expt_config_name = os.path.join("config", "expt_config", f"{expt_config_name}.yaml")
    expt_config = load_config(expt_config_name)

    if expt_config["device"] == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = expt_config["gpu"]
    
    if expt_config["model"] == None:
        raise Exception("Model not specified in expt config file")

    model_config_path = os.path.join("config", "model_config", expt_config["model"] + ".yaml")
    model_config = load_config(model_config_path)

    model_config["device"] = expt_config["device"]
    model_config["data"]["source"] = expt_config["src_data"]
    model_config["data"]["target"] = expt_config["tgt_data"]

    model_config["log"] = expt_config["log"]
    model_config["train"]["log_loss"] = expt_config["log_loss"]
    model_config["verbose"] = expt_config["verbose"]

    model_config["wandb"] = expt_config["wandb"]
    model_config["wandb_project"] = expt_config["wandb_project"]

    if "patience" in model_config["train"]:
        model_config["train"]["early_stopping"] = True
    else:
        model_config["train"]["early_stopping"] = False
    
    if model_config["verbose"]:
        print("\nExperiment Config:\n")
        print(expt_config)
        print("\nModel Config:\n")
        print(model_config)   

    return model_config


def parse_config():
    parser = argparse.ArgumentParser(description="GDA")
    parser.add_argument("--config", type=str, default="dgda_expt", 
                        help="The configuration of the experiment")
    args = parser.parse_args()
    
    configs = build_config(args.config)
    return configs
