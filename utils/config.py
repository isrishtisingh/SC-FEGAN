import argparse
import yaml
import os
import logging
from yaml import safe_load

logger = logging.getLogger()

class Config(object):
    # def __init__(self, filename=None): #OLD
    def __init__(self, filename):
        assert os.path.exists(filename), "ERROR: Config File doesn't exist."
        try:
            with open(filename, 'r') as f:
                self._cfg_dict = safe_load(f)
                # self._cfg_dict = yaml.load(f) #OLD
        # parent of IOError, OSError *and* WindowsError where available
        except EnvironmentError:
            logger.error('Please check the file with name of "%s"', filename)
        logger.info(' APP CONFIG '.center(80, '-'))
        logger.info(''.center(80, '-'))

    def __getattr__(self, name):
        value = self._cfg_dict[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value