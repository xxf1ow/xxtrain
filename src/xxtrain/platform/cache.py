import os
import shutil
from pathlib import Path

from xxtrain.business_tasks import point_detection_recipe
from xxtrain.pipeline import ConversionReport, convert_dataset
from xxtrain.workspace_data import WorkspaceData


def build_detection_cache(data: WorkspaceData, destination: Path) -> ConversionReport:
    """Build and atomically publish the Point detector dataset for completed workspace samples."""
    destination = Path(destination)
    temporary = destination.with_name(f'.{destination.name}.building')
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        source_root = data.materialize_detection_source(temporary)
        report = convert_dataset(point_detection_recipe(), source_root, reserve_no_label=True)
        os.replace(temporary, destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
