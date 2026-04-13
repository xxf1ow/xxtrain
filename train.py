import argparse
import os

import ultralytics
from ruamel.yaml import YAML
from ultralytics.models import YOLO

from annconverter import voc2yolo


def get_template_name(model_version='v8', task_type='detect'):
    suffix_switcher = {'classify': '-cls', 'detect': '', 'obb': '-obb', 'pose': '-pose', 'segment': '-seg'}
    suffix = suffix_switcher.get(task_type)
    if suffix is None:
        return None
    return f'yolo{model_version}{suffix}.yaml'


def get_model_name(model_version='v8', model_scale='n', task_type='detect'):
    if model_scale not in ['n', 's', 'm', 'l', 'x']:
        return None
    suffix_switcher = {'classify': '-cls', 'detect': '', 'obb': '-obb', 'pose': '-pose', 'segment': '-seg'}
    suffix = suffix_switcher.get(task_type)
    if suffix is None:
        return None
    return f'yolo{model_version}{model_scale}{suffix}'


def get_dataset_yaml_path(root_path):
    return os.path.join(root_path, 'dataset.yaml')


def get_model_yaml_path(root_path, model_name):
    return os.path.join(root_path, f'{model_name}.yaml')


def get_pretrained_weights_path(root_path, model_name):
    return os.path.join(root_path, '.weights', f'{model_name}.pt')


def convert_voc_to_yolo(root_path, split, reserve_no_label):
    dataset_yaml_path = get_dataset_yaml_path(root_path)
    yolo_dataset_path = os.path.join(root_path, 'labels')
    if os.path.isfile(dataset_yaml_path) and os.path.isdir(yolo_dataset_path) and os.listdir(yolo_dataset_path):
        print(f'✅ Dataset configuration file already exists at {dataset_yaml_path}\n')
        return
    print('🚀 Converting VOC dataset to YOLO format ...')
    voc2yolo.process(root_path, split, reserve_no_label)
    if not os.path.isfile(dataset_yaml_path):
        print(f'❌ dataset.yaml not found at {dataset_yaml_path} after conversion, please check the process')
        return
    print('✅ COCO conversion done!\n')


def generate_model_yaml(root_path, model_version='v8', model_scale='n', task_type='detect'):
    model_name = get_model_name(model_version, model_scale, task_type)
    if model_name is None:
        raise ValueError(f'❌ Invalid version, scale or task type: {model_version}, {model_scale}, {task_type}')
    target_path = get_model_yaml_path(root_path, model_name)
    print(f'🚀 Generating model YAML for task: {task_type} ...')
    try:
        yaml_handler = YAML()
        yaml_handler.preserve_quotes = True  # 保留引号
        # read num_classes from dataset.yaml
        dataset_yaml_path = get_dataset_yaml_path(root_path)
        with open(dataset_yaml_path) as f:
            num_classes = len(yaml_handler.load(f)['names'])
        if num_classes <= 0:
            raise ValueError(f'❌ No classes found in dataset.yaml: {dataset_yaml_path}')
        # source template model.yaml
        template_name = get_template_name(model_version, task_type)
        if template_name is None:
            raise ValueError(f'❌ Invalid version or task type: {model_version}, {task_type}')
        package_path = os.path.dirname(ultralytics.__file__)
        source_path = os.path.join(package_path, 'cfg', 'models', model_version, template_name)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f'❌ Template model configuration file not found: {source_path}')
        with open(source_path) as f:
            content = yaml_handler.load(f)
            content['nc'] = num_classes
        # target model.yaml
        with open(target_path, 'w') as f:
            yaml_handler.dump(content, f)
    except Exception as e:
        print(f'❌ Failed to generate model configuration file: {e}')
        return
    print(f'✅ Model configuration file generated: {target_path}\n')
    return model_name


def download_pretrained(root_path, model_name):
    # download pretrained weights if not exist
    pretrained_weights = f'{model_name}.pt'
    pretrained_weights_path = get_pretrained_weights_path(root_path, model_name)
    if os.path.isfile(pretrained_weights_path):
        print(f'✅ Pretrained weights already exists at {pretrained_weights_path}\n')
        return
    print(f'🚀 Downloading pretrained weights: {pretrained_weights} ...')
    os.makedirs(os.path.dirname(pretrained_weights_path), exist_ok=True)
    model = YOLO(pretrained_weights)
    model.save(pretrained_weights_path)
    print(f'✅ Pretrained weights downloaded: {pretrained_weights_path}\n')


def process(root_path, model_version='v8', model_scale='n', task_type='detect', split=10, reserve_no_label=True):
    # process dataset and generate dataset.yaml, generate model.yaml, download pretrained weights
    convert_voc_to_yolo(root_path, split, reserve_no_label)
    model_name = generate_model_yaml(root_path, model_version, model_scale, task_type)
    download_pretrained(root_path, model_name)

    # Train the model
    print(f'🚀 Starting training for model: {model_name} ...')
    model = YOLO(get_model_yaml_path(root_path, model_name))
    model.load(get_pretrained_weights_path(root_path, model_name))
    model.train(data=get_dataset_yaml_path(root_path), epochs=100, batch=16, imgsz=640, device=0)
    best_model_path = model.trainer.best if model.trainer and hasattr(model.trainer, 'best') else 'N/A'
    print(f'✅ Training completed! Best model saved at: {best_model_path}\n')
    if not os.path.isfile(best_model_path):
        return

    # Exporting model to ONNX and Optimize the ONNX model using onnxsim
    try:
        import onnx
        import onnxsim

        onnx_path = os.path.join(root_path, f'{model_name}.onnx')
        print('🚀 Exporting best model to ONNX format ...')
        model = YOLO(best_model_path)
        model.export(format='onnx', path=onnx_path)
        print(f'✅ Model exported to ONNX format: {onnx_path}')
        onnxsim_path = os.path.join(root_path, f'{model_name}.sim.onnx')
        model_simplified, check = onnxsim.simplify(onnx.load(onnx_path))
        if check:
            onnx.save(model_simplified, onnxsim_path)
        print(f'✅ Simplified ONNX model saved to: {onnxsim_path}\n')
    except Exception as e:
        print(f'❌ Failed to export model to ONNX format: {e}')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help='Path to VOC dataset root')
    parser.add_argument('--model_version', type=str, default='v8', help='YOLO model version (e.g. v8)')
    parser.add_argument('--model_scale', type=str, default='n', help='YOLO model scale (e.g. n, s, m, l, x)')
    parser.add_argument('--task_type', type=str, default='detect', help='Task type (e.g. detect)')
    parser.add_argument('--split', type=int, default=10, help='Split ratio for test set')
    parser.add_argument('--reserve_no_label', action='store_true', help='Whether to keep images without labels')
    args = parser.parse_args()
    process(args.root_path, args.model_version, args.model_scale, args.task_type, args.split, args.reserve_no_label)
