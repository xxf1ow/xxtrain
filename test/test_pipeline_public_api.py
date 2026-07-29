import unittest

import xxtrain.pipeline as pipeline


class PipelinePublicApiTest(unittest.TestCase):
    def test_package_exports_only_stable_public_symbols(self) -> None:
        self.assertEqual(
            [
                'Context',
                'ConversionConfig',
                'ConversionReport',
                'DatasetRecipe',
                'ExpandProcessor',
                'ImageRef',
                'ItemProcessor',
                'Pipeline',
                'Sample',
                'convert_dataset',
                'standard_recipe',
            ],
            pipeline.__all__,
        )

    def test_package_does_not_bind_internal_symbols(self) -> None:
        for name in (
            'CocoSink',
            'CocoSource',
            'LabelImgSink',
            'LabelMeSink',
            'ExtractSample',
            'ReadAnnotations',
            'ValidateObb',
            'validate_annotations_for_task',
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(pipeline, name))


if __name__ == '__main__':
    unittest.main()
