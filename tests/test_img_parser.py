import unittest
from our_code.ImgParser import Parser


class TestScanPersonFolder(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = Parser('images')

    def test_number_of_photos(self) -> None:
        self.assertEqual(1, len(self.parser.scan_person_folder('images/adi')))
        self.assertEqual(2, len(self.parser.scan_person_folder('images/yaniv')))

    def test_none_picture_files_are_not_counted(self):
        self.assertEqual(1, len(self.parser.scan_person_folder('images/obama')))


class TestGelAllPeople(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = Parser('images')

    def test_number_of_people(self):
        self.assertEqual(6, len(self.parser.get_all_people()))

    def test_names(self):
        self.assertIn("yaniv", self.parser.get_all_people(), "yaniv should be one of the dict keys")

    def test_value_size(self):
        self.assertEqual(2, len(self.parser.get_all_people()['yaniv']), "yaniv have two photos")
