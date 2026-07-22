import unittest

from test.support.current_api import (
    Annotation,
    AnnotationsConverter,
    ImageSizeParser,
    Pipeline,
    ShapeType,
    TaskPayload,
)


class PipelineContractsTest(unittest.TestCase):
    def test_annotations_converter_only_relabels_matching_inputs(self) -> None:
        payload = TaskPayload()
        annotations = {
            'matching': Annotation(label='source', type=ShapeType.RECTANGLE),
            'other': Annotation(label='other', type=ShapeType.RECTANGLE),
        }
        payload.set('det_anns', annotations, 'test')
        pipeline = Pipeline([AnnotationsConverter(in_labels=['source'], out_labels='target')])

        pipeline.process(None, payload)

        self.assertEqual('target', annotations['matching'].label)
        self.assertEqual('other', annotations['other'].label)

    def test_pipeline_validates_processor_required_inputs(self) -> None:
        pipeline = Pipeline([ImageSizeParser()])

        with self.assertRaisesRegex(ValueError, r"\[ImageSizeParser\].*'in_img_path'"):
            pipeline.process(None, TaskPayload())


if __name__ == '__main__':
    unittest.main()
