# encoding: UTF-8

import yaml
import inputs


class Configuration(object):
    def __init__(self, filename):
        # TODO improve UX
        fd = open(filename, encoding='UTF-8')
        self.__values__ = yaml.load(fd)
        self.data_loader = self.__values__.get('data_loader')
        if self.data_loader is None:
            self.data_loader = inputs.get_data_loader(self.dataset_name)
        else:
            # TODO add support for custom data_loader
            raise NotImplementedError('Custom data loader not supported yet')

    def __getattr__(self, item):
        return self.__values__[item]
