import os

__file_dir__ = os.path.dirname(os.path.realpath(__file__))
__data_dir__ = f'{__file_dir__}/Datasets/'
__resources_dir__ = f'{__file_dir__}/resources/'

if __name__=='__main__':
    print(__data_dir__)