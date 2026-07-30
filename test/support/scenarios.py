from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from xxtrain.training import TrainingScenario, load_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_PATHS = {
    'detect': PROJECT_ROOT / 'data/standard-detect/standard_detect.py',
    'segment': PROJECT_ROOT / 'data/standard-segment/standard_segment.py',
    'pose': PROJECT_ROOT / 'data/standard-pose/standard_pose.py',
    'classify': PROJECT_ROOT / 'data/standard-classify/standard_classify.py',
    'direction-sensitive-classify': (PROJECT_ROOT / 'data/standard-classify/direction_sensitive_classify.py'),
    'tuned-classify': PROJECT_ROOT / 'data/standard-classify/tuned_classify.py',
    'point-detect': PROJECT_ROOT / 'data/point/point_detect.py',
    'point-classify': PROJECT_ROOT / 'data/point/point_classify.py',
    'point-segment': PROJECT_ROOT / 'data/point/point_segment.py',
    'knob-detect': PROJECT_ROOT / 'data/knob/knob_detect.py',
    'knob-segment': PROJECT_ROOT / 'data/knob/knob_segment.py',
    'light1-detect': PROJECT_ROOT / 'data/light/light1_detect.py',
    'light2-detect': PROJECT_ROOT / 'data/light/light2_detect.py',
}


def load_case_scenario(name: str) -> TrainingScenario:
    return load_scenario(SCENARIO_PATHS[name])


def materialization_only(task_type: str):
    return (
        patch('xxtrain.pipeline.workflow._validate_output_labels') if task_type == 'point-classify' else nullcontext()
    )
