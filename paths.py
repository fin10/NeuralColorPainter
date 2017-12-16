import os


class Paths:
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    MODEL = os.path.join(ROOT, 'model')
    IMAGES = os.path.join(ROOT, 'images')
    OUTPUT = os.path.join(ROOT, 'out')
