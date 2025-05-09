import logging
import sys


def _init_logger() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # add stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    logging.info("init logging")


_init_logger()
