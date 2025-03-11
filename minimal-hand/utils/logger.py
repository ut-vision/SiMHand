import os
import logging


def setup_logger(log_file_path, resume = False):

    if os.path.exists(log_file_path) and resume == 'False':
        open(log_file_path, 'w').close()
        

    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)


    logging.getLogger().addHandler(console_handler)

    return logging.getLogger()