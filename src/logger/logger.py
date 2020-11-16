import logging.config
import json

from src.utils.path_utils import get_app_resources_path


with open(get_app_resources_path().joinpath('logging.json'), 'r+') as file:
    logging_config = json.load(file)

logging.basicConfig()
logging.config.dictConfig(logging_config)


def get_logger():
    '''
     method read the logging configurations (.json) construct the logging object
    :return: logging object to log the messages
    '''
    logger = logging.getLogger(__name__)
    return logger
