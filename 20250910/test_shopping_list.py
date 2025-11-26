import unittest

# python -m unittest
import shopping_list
from shopping_list import ShoppingList

class TestShoppingList(unittest.TestCase):
    def setUp(self):
        self.shopping_list = ShoppingList({"牙刷": 5, "沐浴露": 15, "电池": 7})

    def test_get_item_count(self):
        self.assertEqual(self.shopping_list.get_item_count(),3)

    def test_get_total_price(self):
        self.assertEqual(self.shopping_list.get_total_price(),27)