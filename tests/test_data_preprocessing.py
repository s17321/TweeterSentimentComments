# tests/test_data_preprocessing.py

import unittest
from src.data_preprocessing import clean_text

class TestDataPreprocessing(unittest.TestCase):
    def test_clean_text(self):
        text = "I love machine learning! Visit http://example.com #AI"
        expected = "love machine learning ai"
        self.assertEqual(clean_text(text), expected)
        
    def test_clean_text_empty(self):
        text = "!!! ###"
        expected = ""
        self.assertEqual(clean_text(text), expected)

if __name__ == '__main__':
    unittest.main()
