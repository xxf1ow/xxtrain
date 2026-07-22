import shutil
import tempfile
import unittest
from pathlib import Path

from test.support.current_api import TaskType, convert_dataset, parse_labelimg, parse_labelme

FIXTURES_PATH = Path(__file__).resolve().parent / 'fixtures'


class FailureBehaviorTest(unittest.TestCase):
    def copy_fixture(self, fixture_name: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix='xxtrain-failures-')
        self.addCleanup(temp_dir.cleanup)
        root_path = Path(temp_dir.name) / fixture_name
        shutil.copytree(FIXTURES_PATH / fixture_name, root_path)
        return root_path

    def test_labelimg_rejects_mismatched_image_width(self) -> None:
        root_path = self.copy_fixture('standard-detect')
        annotation_path = root_path / 'src' / '20260620' / 'anns' / '0000.xml'

        with self.assertRaisesRegex(Exception, '图片与标签不对应'):
            parse_labelimg(str(annotation_path), 1919, 1080)

    def test_labelme_rejects_mismatched_image_width(self) -> None:
        root_path = self.copy_fixture('standard-segment')
        annotation_path = root_path / 'src' / '251010' / 'anns_seg' / '0000.json'

        with self.assertRaisesRegex(Exception, '图片与标签不对应'):
            parse_labelme(str(annotation_path), 1919, 1080, TaskType.SEGMENT)

    def test_invalid_annotation_reports_parse_failure(self) -> None:
        root_path = self.copy_fixture('standard-segment')
        annotation_path = root_path / 'src' / '251010' / 'anns_seg' / 'invalid.json'
        annotation_path.write_text('{', encoding='utf-8')

        with self.assertRaisesRegex(Exception, 'Failed to parse annotation'):
            parse_labelme(str(annotation_path), 1920, 1080, TaskType.SEGMENT)

    def test_unknown_task_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unsupported task type: unknown'):
            convert_dataset('unknown', 'unused', split=10, reserve_no_label=False)

    def test_classification_rejects_extra_class_directory(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        (root_path / 'src' / 'extra' / 'imgs').mkdir(parents=True)

        with self.assertRaisesRegex(ValueError, 'Classification labels mismatch'):
            convert_dataset('classify', str(root_path), split=10, reserve_no_label=False)

    def test_classification_rejects_split_without_train_samples(self) -> None:
        root_path = self.copy_fixture('standard-classify')
        for class_path in (root_path / 'src').iterdir():
            images_path = class_path / 'imgs'
            if not images_path.is_dir():
                continue
            for image_path in sorted(images_path.iterdir())[1:]:
                image_path.unlink()

        with self.assertRaisesRegex(ValueError, 'without train or val samples'):
            convert_dataset('classify', str(root_path), split=10, reserve_no_label=False)


if __name__ == '__main__':
    unittest.main()
