from config.configurator import parse_config
from trainer.utils import init_seed
from trainer.logger import Logger, WandbHandler

from models.build_model import build_model
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer

def train():
    print(configs.get("a", 0))
    print("Running the experiment")
    print("Done running the experiment")
    return True

def test():
    pass

def tune():
    pass

if __name__ == "__main__":

    configs = parse_config()
    init_seed(configs)
    
    data_handler = build_data_handler(configs)
    data_handler.load_data()
    model = build_model(data_handler, configs).to(configs["device"])

    logger = Logger(configs)
    trainer = build_trainer(data_handler, logger, configs)