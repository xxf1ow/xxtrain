import argparse
import os
import shutil
from datetime import datetime

import ultralytics
from ruamel.yaml import YAML
from tqdm import tqdm
from ultralytics.engine.results import Probs
from ultralytics.models import YOLO

from xxtrain.pipeline import convert_dataset

suffix_switcher = {'classify': '-cls', 'detect': '', 'obb': '-obb', 'pose': '-pose', 'segment': '-seg'}


def get_template_name(model_version: str = 'v8', task_type: str = 'detect'):
    for task, suffix in suffix_switcher.items():
        if task_type.endswith(task):
            return f'yolo{model_version}{suffix}.yaml'
    return None


def get_model_name(model_version: str = 'v8', model_scale: str = 'n', task_type: str = 'detect'):
    if model_scale not in ['n', 's', 'm', 'l', 'x']:
        return None
    for task, suffix in suffix_switcher.items():
        if task_type.endswith(task):
            return f'yolo{model_version}{model_scale}{suffix}'
    return None


def get_dataset_yaml_path(root_path: str, task_type: str):
    return os.path.join(root_path, task_type, 'dataset.yaml')


def get_model_yaml_path(root_path: str, task_type: str, model_name: str):
    return os.path.join(root_path, task_type, f'{model_name}.yaml')


def get_pretrained_weights_path(model_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, '.weights', f'{model_name}.pt')


def convert_voc_to_yolo(task_type: str, root_path: str, split: int, reserve_no_label: bool):
    dataset_yaml_path = get_dataset_yaml_path(root_path, task_type)
    if os.path.isfile(dataset_yaml_path) or (task_type.endswith('classify') and os.path.isdir(dataset_yaml_path)):
        print(f'✅ Dataset configuration file already exists at {dataset_yaml_path}\n')
        return
    print('🚀 Converting dataset to YOLO format ...')
    convert_dataset(
        task_type,
        root_path,
        split=split,
        reserve_no_label=reserve_no_label,
    )
    if os.path.isfile(dataset_yaml_path) or (task_type.endswith('classify') and os.path.isdir(dataset_yaml_path)):
        print('✅ Conversion done!\n')
        return
    print(f'❌ dataset.yaml not found at {dataset_yaml_path} after conversion, please check the process')


def generate_model_yaml(root_path: str, model_version: str = 'v8', model_scale: str = 'n', task_type: str = 'detect'):
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


def download_pretrained(model_name: str):
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


def train_model(root_path: str, model_name: str, task_type: str) -> YOLO:
    # Train the model
    print(f'🚀 Starting training for model: {model_name} ...')
    model = YOLO(get_model_yaml_path(root_path, task_type, model_name))
    model.load(get_pretrained_weights_path(model_name))

    if not task_type.endswith('classify'):
        model.train(data=get_dataset_yaml_path(root_path, task_type), epochs=100, batch=32, imgsz=640)
    else:
        if not task_type.startswith('point') and not task_type.startswith('knob'):
            model.train(data=os.path.join(root_path, task_type), epochs=72, batch=128, imgsz=224)
        else:
            model.train(
                data=os.path.join(root_path, task_type),
                epochs=72,
                batch=128,
                imgsz=224,
                fliplr=0.0,  # 针对方向敏感型数据集: 禁止左右翻转
                flipud=0.0,  # 针对方向敏感型数据集: 禁止上下翻转
                degrees=0.0,  # 针对方向敏感型数据集: 禁止旋转
                auto_augment=None,  # 针对方向敏感型数据集: 禁止自动数据增强
            )

    best_model_path = model.trainer.best if model.trainer and hasattr(model.trainer, 'best') else ''
    if not os.path.isfile(best_model_path):
        print(f'❌ Training completed! But the best model checkpoint not found at {best_model_path} ...')
        return model
    print(f'✅ Training completed! Best model saved at: {best_model_path}\n')
    model = YOLO(best_model_path)
    return model


def copy_class_reference_images(root_path: str, task_type: str, onnx_path: str, class_names: dict[int, str]):
    references = []
    train_path = os.path.join(root_path, task_type, 'train')
    for class_index, class_name in sorted(class_names.items()):
        class_path = os.path.join(train_path, class_name)
        with os.scandir(class_path) as image_entries:
            image_path = next((entry.path for entry in image_entries if entry.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f'No reference image found for class: {class_name}')
        references.append((class_index, class_name, image_path))
    references_path = f'{os.path.splitext(onnx_path)[0]}_references'
    os.makedirs(references_path, exist_ok=True)
    for class_index, class_name, image_path in references:
        extension = os.path.splitext(image_path)[1]
        shutil.copy(image_path, os.path.join(references_path, f'{class_index}_{class_name}{extension}'))


def export_model_to_onnx(best_model: YOLO, root_path: str, model_name: str):
    # Exporting model to ONNX and Optimize the ONNX model using onnxsim
    try:
        print('🚀 Exporting best model to ONNX format ...')
        temp_onnx_path = best_model.export(format='onnx', simplify=True)
        formatted_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        onnx_path = os.path.join(root_path, 'weights', f'{model_name}_{formatted_time}.onnx')
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        shutil.move(temp_onnx_path, onnx_path)
        print(f'✅ Model exported to ONNX format: {onnx_path}')
        return onnx_path
    except Exception as e:
        print(f'❌ Failed to export model to ONNX format: {e}')
        return


def standard_validate(best_model: YOLO, directory: str):
    # Validate the model using the best checkpoint
    print('🚀 Running inference on validation set using best model ...')
    results = best_model.predict(source=directory, verbose=False, save=True, stream=True)
    if isinstance(results, map) or hasattr(results, '__iter__'):
        for _ in results:
            pass
    assert best_model.predictor and hasattr(best_model.predictor, 'save_dir')
    print(f'✅ Inference completed. Results are saved in {best_model.predictor.save_dir}')


def classify_validate(best_model: YOLO, directory: str):
    # Validate the model using the best checkpoint
    assert model.task == 'classify'
    class_names = best_model.names
    name_to_idx = {v: k for k, v in class_names.items()}
    if not class_names:
        print('⚠️ 标签列表是空的 ...')
        return

    # 递归得到所有图片文件路径
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    abs_root_path = os.path.abspath(directory)
    image_paths = []
    for root, dirs, files in os.walk(abs_root_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    if not image_paths:
        print(f'⚠️ 在目录 {directory} 下未找到任何图片文件。')
        return

    # run
    print(f'🚀 Running inference using best model, image size: {len(image_paths)} ...')
    total_count = 0
    mismatched = []
    for imgpath in tqdm(image_paths, leave=True, colour='CYAN'):
        results = best_model.predict(source=imgpath, verbose=False, save=False)
        res = results[0]
        total_count += 1
        true_name = os.path.basename(os.path.dirname(imgpath))
        true_idx = name_to_idx.get(true_name)
        assert true_idx is not None
        assert isinstance(res.probs, Probs)
        pred_idx = res.probs.top1
        pred_name = class_names[pred_idx]
        if pred_idx != true_idx:
            assert best_model.predictor and hasattr(best_model.predictor, 'save_dir')
            save_dir = os.path.join(best_model.predictor.save_dir, f'{true_name} - {pred_name}')
            os.makedirs(save_dir, exist_ok=True)
            shutil.copy(imgpath, os.path.join(save_dir, os.path.basename(imgpath)))
            mismatched.append(f'expect: {true_idx}-{true_name}, actual: {pred_idx}-{pred_name} ==> {imgpath}')

    # 写入到文件
    print('✅ Validation completed!')
    print(f'📊 Total samples processed: {total_count}')
    if len(mismatched) > 0:
        assert best_model.predictor and hasattr(best_model.predictor, 'save_dir')
        mismatched_txt = os.path.join(best_model.predictor.save_dir, 'mismatched_samples.txt')
        with open(mismatched_txt, 'w', encoding='utf-8') as f:
            f.write('# Mismatched Samples Report\n')
            f.write(f'# Total mismatched: {len(mismatched)} / {total_count}\n')
            f.write('\n'.join(mismatched))
        print(f'❌ Found {len(mismatched)} mismatched samples.')
        print(f'📄 Results saved at: {mismatched_txt}\n')


def process(
    root_path: str,
    model_version: str = 'v8',
    model_scale: str = 'n',
    task_type: str = 'detect',
    split: int = 10,
    reserve_no_label: bool = True,
):
    # process dataset and generate dataset.yaml, generate model.yaml, download pretrained weights
    convert_voc_to_yolo(task_type, root_path, split, reserve_no_label)
    model_name = generate_model_yaml(root_path, model_version, model_scale, task_type)
    download_pretrained(model_name)
    best_model = train_model(root_path, model_name, task_type)
    onnx_path = export_model_to_onnx(best_model, root_path, model_name)
    if onnx_path and task_type.endswith('classify'):
        copy_class_reference_images(root_path, task_type, onnx_path, best_model.names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Path to VOC dataset root')
    parser.add_argument('--model_version', type=str, default='v8', help='YOLO model version (e.g. v8)')
    parser.add_argument('--model_scale', type=str, default='n', help='YOLO model scale (e.g. n, s, m, l, x)')
    parser.add_argument('--task_type', type=str, default='detect', help='Task type (e.g. detect)')
    parser.add_argument('--split', type=int, default=10, help='Split ratio for test set')
    parser.add_argument('--reserve_no_label', action='store_true', help='Whether to keep images without labels')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'export', 'val'], help='Run mode')
    parser.add_argument('--weights', type=str, help='Path to .pt model weights (required for export/val)')
    parser.add_argument('--directory', type=str, help='Path to images directory (required for export/val)')
    args = parser.parse_args()

    model_name = get_model_name(args.model_version, args.model_scale, args.task_type)
    assert model_name is not None, (
        f'❌ Invalid version, scale or task type: {args.model_version}, {args.model_scale}, {args.task_type}'
    )

    if args.mode == 'train':
        process(args.root_path, args.model_version, args.model_scale, args.task_type, args.split, args.reserve_no_label)

    elif args.mode == 'export':
        """
        python3 src/train.py \
            --mode export \
            --root_path data/point \
            --task_type point-classify \
            --weights runs/classify/train9/weights/best.pt
        """
        if not args.weights:
            print('❌ Error: --weights is required for export mode')
        else:
            print(f'🚀 Loading model for export: {args.weights}')
            model = YOLO(args.weights)
            onnx_path = export_model_to_onnx(model, args.root_path, model_name)
            if onnx_path and args.task_type.endswith('classify'):
                copy_class_reference_images(args.root_path, args.task_type, onnx_path, model.names)

    elif args.mode == 'val':
        """
        python3 src/train.py --mode val \
        --directory /home/lxx/ultralytics/xxtrain/data/light/light-classify \
        --weights runs/classify/train16/weights/best.pt
        """
        if not args.weights or not args.directory:
            print('❌ Error: --weights and --directory is required for val mode')
        else:
            model = YOLO(args.weights)
            print(f'🚀 Loading model for validation: {args.weights}, task type: {model.task} ...')
            if model.task == 'classify':
                classify_validate(model, args.directory)
            else:
                standard_validate(model, args.directory)
