import os
import shutil
from pathlib import Path

import yaml

from xxtrain.business_tasks import point_detection_recipe
from xxtrain.data.dataset import write_split_lists
from xxtrain.pipeline import ConversionReport, convert_dataset
from xxtrain.workspace_data import WorkspaceData


def build_detection_cache(data: WorkspaceData, destination: Path) -> ConversionReport:
    """Build and atomically publish the Point detector dataset for completed workspace samples."""
    destination = Path(destination).absolute()
    temporary = destination.with_name(f'.{destination.name}.building')
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        source_root = data.materialize_detection_source(temporary)
        report = convert_dataset(point_detection_recipe(), source_root, reserve_no_label=True)
        # References must survive publication of the entire build directory by rename.
        for image in (temporary / 'detect' / 'workspace').iterdir():
            if image.is_symlink():
                target = os.path.relpath(image.resolve(), image.parent)
                image.unlink()
                image.symlink_to(target)
        report.train_items = [str(destination / Path(item).relative_to(temporary)) for item in report.train_items]
        report.val_items = [str(destination / Path(item).relative_to(temporary)) for item in report.val_items]
        write_split_lists(temporary, 'detect', report.train_items, report.val_items)
        dataset_path = temporary / 'detect' / 'dataset.yaml'
        dataset = yaml.safe_load(dataset_path.read_text(encoding='utf-8'))
        dataset['path'] = str(destination)
        dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding='utf-8')
        os.replace(temporary, destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
