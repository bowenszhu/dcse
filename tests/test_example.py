import unittest
from src.example import example_func


class TestStringMethods(unittest.TestCase):

    def test_addition(self):
        # A simple example of test
        self.assertEqual(1 + 1, 2)

    def test_foo(self):
        self.assertEqual(2, example_func(1))


if __name__ == '__main__':
    unittest.main()
