import logging

def get_logger(logging_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
       "%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s"
    )

    file_log_handler = logging.FileHandler(logging_file, mode = 'w')
    file_log_handler.setLevel(logging.INFO)
    file_log_handler.setFormatter(formatter)

    #  stream_log_handler = logging.StreamHandler()
    #  stream_log_handler.setLevel(logging.INFO)
    #  stream_log_handler.setFormatter(formatter)

    logger.addHandler(file_log_handler)
    #  logger.addHandler(stream_log_handler)

    return logger 