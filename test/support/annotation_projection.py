from xxtrain.data import Bbox, Pose


def project_annotations(annotations, info):
    group_members = {}
    for index, annotation in enumerate(annotations):
        if annotation.group is not None:
            group_members.setdefault(annotation.group, []).append(index)
    normalized_groups = {
        group: f'group-{index}'
        for index, group in enumerate(group for group, members in group_members.items() if len(members) > 1)
    }

    def point(x, y):
        return round(x / info.width, 6), round(y / info.height, 6)

    def project(annotation):
        common = {
            'type': annotation.type.value,
            'label': annotation.label,
            'group': normalized_groups.get(annotation.group),
        }
        if isinstance(annotation, Pose):
            return common | {
                'bbox': (*point(annotation.x1, annotation.y1), *point(annotation.x2, annotation.y2)),
                'keypoints': tuple(
                    (keypoint.label, *point(keypoint.x, keypoint.y), keypoint.visibility)
                    for keypoint in annotation.keypoints
                ),
            }
        if isinstance(annotation, Bbox):
            return common | {'bbox': (*point(annotation.x1, annotation.y1), *point(annotation.x2, annotation.y2))}
        return common | {'points': tuple(point(x, y) for x, y in annotation.points)}

    return tuple(project(annotation) for annotation in annotations)
