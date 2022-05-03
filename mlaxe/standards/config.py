"""
This module stores config

"""


class Config:
    colors = list('bgrcmyk') + ['purple', 'lime', 'olive', 'pink',
                                'coral', 'indigo', 'gold', 'chocolate']

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
