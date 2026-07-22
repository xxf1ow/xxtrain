import unittest

from xxtrain.data import LabelCatalog


class LabelCatalogTest(unittest.TestCase):
    def test_preserves_order_and_numeric_names_as_strings(self) -> None:
        labels = LabelCatalog(names=('switch', '1008', '0'))

        self.assertEqual(('switch', '1008', '0'), tuple(labels))
        self.assertEqual(1, labels.index('1008'))
        self.assertIn('0', labels)
        self.assertEqual(3, len(labels))

    def test_unknown_name_uses_value_error(self) -> None:
        with self.assertRaises(ValueError):
            LabelCatalog(names=('known',)).index('unknown')

    def test_rejects_non_tuple_name_containers(self) -> None:
        with self.assertRaises(ValueError):
            LabelCatalog(names=['a'])

    def test_rejects_empty_catalog_names_and_duplicates(self) -> None:
        for names in ((), ('',), ('a', 'a'), ('a', 1)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                LabelCatalog(names=names)


if __name__ == '__main__':
    unittest.main()
