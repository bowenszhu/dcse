import unittest
from src.example import example_func


class TestStringMethods(unittest.TestCase):

    def test_example_func(self):
        self.assertEqual(2, example_func(1))


if __name__ == '__main__':
    unittest.main()
