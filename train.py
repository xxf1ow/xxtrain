import argparse
import os
import shutil
from datetime import datetime

import ultralytics
from ruamel.yaml import YAML
from ultralytics.models import YOLO

import annconverter

suffix_switcher = {'classify': '-cls', 'detect': '', 'obb': '-obb', 'pose': '-pose', 'segment': '-seg'}


def get_template_name(model_version='v8', task_type='detect'):
    for task, suffix in suffix_switcher.items():
        if task_type.endswith(task):
            return f'yolo{model_version}{suffix}.yaml'
    return None


def get_model_name(model_version='v8', model_scale='n', task_type='detect'):
    if model_scale not in ['n', 's', 'm', 'l', 'x']:
        return None
    for task, suffix in suffix_switcher.items():
        if task_type.endswith(task):
            return f'yolo{model_version}{model_scale}{suffix}'
    return None


def get_dataset_yaml_path(root_path, task_type):
    return os.path.join(root_path, task_type, 'dataset.yaml')


def get_train_dataset(root_path, task_type):
    if not task_type.endswith('classify'):
        return get_dataset_yaml_path(root_path, task_type)
    return os.path.join(root_path, task_type)


def get_model_yaml_path(root_path, task_type, model_name):
    return os.path.join(root_path, task_type, f'{model_name}.yaml')


def get_pretrained_weights_path(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, '.weights', f'{model_name}.pt')


def convert_voc_to_yolo(task_type, root_path, split, reserve_no_label):
    dataset_yaml_path = get_dataset_yaml_path(root_path, task_type)
    if os.path.isfile(dataset_yaml_path) or (task_type.endswith('classify') and os.path.isdir(dataset_yaml_path)):
        print(f'✅ Dataset configuration file already exists at {dataset_yaml_path}\n')
        return
    print('🚀 Converting dataset to YOLO format ...')
    annconverter.process(task_type, root_path, split, reserve_no_label)
    if os.path.isfile(dataset_yaml_path) or (task_type.endswith('classify') and os.path.isdir(dataset_yaml_path)):
        print('✅ Conversion done!\n')
        return
    print(f'❌ dataset.yaml not found at {dataset_yaml_path} after conversion, please check the process')


def generate_model_yaml(root_path, model_version='v8', model_scale='n', task_type='detect'):
    model_name = get_model_name(model_version, model_scale, task_type)
    if model_name is None:
        raise ValueError(f'❌ Invalid version, scale or task type: {model_version}, {model_scale}, {task_type}')
    target_path = get_model_yaml_path(root_path, task_type, model_name)
    print(f'🚀 Generating model YAML for task: {task_type} ...')
    try:
        yaml_handler = YAML()
        yaml_handler.preserve_quotes = True  # 保留引号

        # source template model.yaml
        template_name = get_template_name(model_version, task_type)
        if template_name is None:
            raise ValueError(f'❌ Invalid version or task type: {model_version}, {task_type}')
        package_path = os.path.dirname(ultralytics.__file__)
        source_path = os.path.join(package_path, 'cfg', 'models', model_version, template_name)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f'❌ Template model configuration file not found: {source_path}')

        # read num_classes from dataset.yaml
        dataset_yaml_path = get_dataset_yaml_path(root_path, task_type)
        with open(dataset_yaml_path) as f:
            dataset = yaml_handler.load(f)

        # read template model.yaml and write target model.yaml
        with open(source_path) as f:
            model = yaml_handler.load(f)

        # set num_classes in model.yaml content
        num_classes = len(dataset['names'])
        if num_classes <= 0:
            raise ValueError(f'❌ No classes found in dataset.yaml: {dataset_yaml_path}')
        model['nc'] = num_classes

        # set kpt_shape in model.yaml content if pose task
        if task_type.endswith('pose'):
            if 'kpt_shape' not in dataset:
                raise KeyError("❌ 'kpt_shape' missing in dataset.yaml for pose task")
            model['kpt_shape'] = dataset['kpt_shape']

        # target model.yaml
        with open(target_path, 'w') as f:
            yaml_handler.dump(model, f)

    except Exception as e:
        raise ValueError(f'❌ Failed to generate model configuration file: {e}')

    print(f'✅ Model configuration file generated: {target_path}\n')
    return model_name


def download_pretrained(model_name):
    # download pretrained weights if not exist
    pretrained_weights = f'{model_name}.pt'
    pretrained_weights_path = get_pretrained_weights_path(model_name)
    if os.path.isfile(pretrained_weights_path):
        print(f'✅ Pretrained weights already exists at {pretrained_weights_path}\n')
        return
    print(f'🚀 Downloading pretrained weights: {pretrained_weights} ...')
    os.makedirs(os.path.dirname(pretrained_weights_path), exist_ok=True)
    model = YOLO(pretrained_weights)
    model.save(pretrained_weights_path)
    print(f'✅ Pretrained weights downloaded: {pretrained_weights_path}\n')


def train_model(root_path, model_name, task_type):
    # Train the model
    print(f'🚀 Starting training for model: {model_name} ...')
    model = YOLO(get_model_yaml_path(root_path, task_type, model_name))
    model.load(get_pretrained_weights_path(model_name))
    batch = 32 if not task_type.endswith('classify') else 128  # 分类任务可以上大 batch
    imgsz = 640 if not task_type.endswith('classify') else 224  # 分类任务不需要大尺寸输入
    model.train(data=get_train_dataset(root_path, task_type), epochs=100, batch=batch, imgsz=imgsz)
    best_model_path = model.trainer.best if model.trainer and hasattr(model.trainer, 'best') else 'N/A'
    print(f'✅ Training completed! Best model saved at: {best_model_path}\n')
    return best_model_path


def validate_model(root_path, model_name, best_model_path):
    # todo: 改为仅分类模型使用, 用来检查数据集中的错误标注或困难样本
    # Validate the model using the best checkpoint
    print(f'🚀 Running inference on validation set using best model: {best_model_path} ...')
    if not os.path.isfile(best_model_path):
        print(f'❌ Best model checkpoint not found at {best_model_path}, skipping ...')
        return
    val_path = os.path.join(root_path, 'val.txt')
    save_path = os.path.abspath(os.path.join(root_path, 'val_results'))
    #  model = YOLO(best_model_path)
    #  model.predict(source=val_path, save=True, conf=0.25, project=save_path, name=model_name, exist_ok=True)
    print(f'✅ Inference completed! Results saved at: {save_path}\n')


def export_model_to_onnx(best_model_path, root_path, model_name):
    # Exporting model to ONNX and Optimize the ONNX model using onnxsim
    try:
        print('🚀 Exporting best model to ONNX format ...')
        model = YOLO(best_model_path)
        temp_onnx_path = model.export(format='onnx', simplify=True)
        formatted_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        onnx_path = os.path.join(root_path, 'weights', f'{model_name}_{formatted_time}.onnx')
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        shutil.move(temp_onnx_path, onnx_path)
        print(f'✅ Model exported to ONNX format: {onnx_path}')
    except Exception as e:
        print(f'❌ Failed to export model to ONNX format: {e}')
        return


def process(
    root_path, model_version='v8', model_scale='n', task_type='detect', split=10, reserve_no_label=True, validate=False
):
    # process dataset and generate dataset.yaml, generate model.yaml, download pretrained weights
    convert_voc_to_yolo(task_type, root_path, split, reserve_no_label)
    model_name = generate_model_yaml(root_path, model_version, model_scale, task_type)
    download_pretrained(model_name)
    best_model_path = train_model(root_path, model_name, task_type)
    if validate:
        validate_model(root_path, model_name, best_model_path)
    export_model_to_onnx(best_model_path, root_path, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help='Path to VOC dataset root')
    parser.add_argument('--model_version', type=str, default='v8', help='YOLO model version (e.g. v8)')
    parser.add_argument('--model_scale', type=str, default='n', help='YOLO model scale (e.g. n, s, m, l, x)')
    parser.add_argument('--task_type', type=str, default='detect', help='Task type (e.g. detect)')
    parser.add_argument('--split', type=int, default=10, help='Split ratio for test set')
    parser.add_argument('--reserve_no_label', action='store_true', help='Whether to keep images without labels')
    parser.add_argument('--validate', type=bool, default=False, help='Whether to run validation after training')
    args = parser.parse_args()

    # # only export ONNX model without training
    # best_model_path = '/home/lxx/ultralytics/xxtrain/runs/pose/train/weights/best.pt'
    # model_name = get_model_name(args.model_version, args.model_scale, args.task_type)
    # export_model_to_onnx(args.root_path, model_name, get_pretrained_weights_path(model_name))

    process(args.root_path, args.model_version, args.model_scale, args.task_type, args.split, args.reserve_no_label)
