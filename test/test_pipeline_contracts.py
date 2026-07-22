import unittest

from test.support.current_api import ImageSizeParser, Pipeline, TaskPayload


class PipelineContractsTest(unittest.TestCase):
    def test_pipeline_validates_processor_required_inputs(self) -> None:
        pipeline = Pipeline([ImageSizeParser()])

        with self.assertRaisesRegex(ValueError, r"\[ImageSizeParser\].*'in_img_path'"):
            pipeline.process(None, TaskPayload())


if __name__ == '__main__':
    unittest.main()
