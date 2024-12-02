import logging
import logging.handlers
import yaml
import os
from logging.config import dictConfig

logger = logging.getLogger(__name__)

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
}
def configure_logging(logfile_path):
    """
    Initialize logging defaults for Project.

    :param logfile_path: logfile used to the logfile
    :type logfile_path: string

    This function does:

    - Assign INFO and DEBUG level to logger file handler and console handler

    """
    dictConfig(DEFAULT_LOGGING)

    logging.root.setLevel(logging.DEBUG)
    logging.root.handlers.clear()

    if logfile_path is not None:
        file_formatter = logging.Formatter(
            "[%(asctime)s, %(levelname)s, %(filename)s:%(funcName)s():%(lineno)s] %(message)s",
            "%d/%m/%y %H:%M:%S")
        file_handler = logging.handlers.RotatingFileHandler(
            logfile_path, maxBytes=10485760, backupCount=300, encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)

    console_formatter = logging.Formatter(
        "[%(levelname)s, %(funcName)s():%(lineno)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logging.root.addHandler(console_handler)


def log_hparams(u_args, c_args, s_args, settings_dir=None):
    if settings_dir is None: settings_dir = u_args['save_path']
    with open(os.path.join(settings_dir, 'settings.yml'), 'w') as yml_file:
        save_dict = {
            'u_args' : u_args,
            'server' : s_args,
            'client' : c_args
        }
        yaml.dump(save_dict, yml_file, default_flow_style=False)