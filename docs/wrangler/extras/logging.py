
import logging
import yaml


def get_default_logging_config():
    """Returns a default logging configuration in dict format"""
    default_config = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'simple'
            },
            # 'file': {
            #     'class': 'logging.FileHandler',
            #     'filename': 'logs.log',
            #     'level': 'DEBUG',
            #     'formatter': 'simple'
            #     }
        },
        'root': {
            "handlers": ['console'],
            'level': 'DEBUG'
        }
    }
    return default_config


def save_default_logging_config(filename):
    """Saves to a .yml file the default configuration file"""
    if not filename.endswith(".yml"):
        filename = filename+".yml"
    config = get_default_logging_config()
    with open(filename,"w") as f:
        yaml.safe_dump(config,f)


def init_default_logging():
    """Sets the default logging configuration"""
    config = get_default_logging_config()
    logging.config.dictConfig(config)


def init_custom_logging(config):
    """Sets the passed config to the logging"""
    logging.config.dictConfig(config)