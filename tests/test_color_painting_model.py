import os
import shutil
import unittest

from color_painting_model import ColorPaintingModel
from paths import Paths


class ColorPaintingModelTest(unittest.TestCase):

    def test_train(self):
        if os.path.exists(Paths.MODEL):
            shutil.rmtree(Paths.MODEL)

        model = ColorPaintingModel()
        model.train()

    def test_color(self):
        if os.path.exists(Paths.OUTPUT):
            shutil.rmtree(Paths.OUTPUT)

        model = ColorPaintingModel()
        model.color()
