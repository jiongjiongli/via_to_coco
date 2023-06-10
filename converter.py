import json
from pathlib import Path

from shapely.geometry import Polygon
from PIL import Image


class CocoFormat:
    @staticmethod
    def create_category_info(category_id, category_name):
        category_info = {'id': category_id,
                         'name': category_name}
        return category_info

    @staticmethod
    def create_image_info(image_id,
                          image_file_name,
                          image_height,
                          image_width):
        image_info = {'id': image_id,
                      'file_name': image_file_name,
                      'height': image_height,
                      'width': image_width}

        return image_info

    @staticmethod
    def create_annotation_info(annotation_id,
                               image_id,
                               category_id,
                               is_crowd,
                               bbox,
                               segmentation,
                               area):
        annotation_info = {'id': annotation_id,
                           'image_id': image_id,
                           'category_id': category_id,
                           'iscrowd': is_crowd,
                           'bbox': bbox,
                           'area': area,
                           'segmentation': segmentation}

        return annotation_info

def convert(image_dir_path,
            via_ann_file_path,
            category_names,
            output_file_path=None,
            first_category_index=1):
    category_dict = {}

    coco_categories = []

    for category_id, category_name in enumerate(category_names, start=first_category_index):
        category_dict[category_name] = category_id
        coco_category = CocoFormat.create_category_info(category_id, category_name)
        coco_categories.append(coco_category)

    with open(Path(via_ann_file_path).as_posix(), 'r') as file_stream:
        via_ann_data = json.load(file_stream)

    coco_images = []
    coco_annotations = []

    image_dir_path = Path(image_dir_path)
    default_category_name = category_names[0]
    image_id = 0
    annotation_id = 0

    for via_ann_key, via_ann in via_ann_data.items():
        image_file_name = via_ann['filename']
        image_file_path = image_dir_path / image_file_name

        with Image.open(image_file_path.as_posix()) as image:
            image_width, image_height = image.size

        image_info = CocoFormat.create_image_info(
                                                  image_id,
                                                  Path(image_file_name).name,
                                                  image_height,
                                                  image_width
        )

        coco_images.append(image_info)

        via_regions = via_ann['regions']

        for via_region_index, via_region in via_regions.items():
            region_attributes = via_region['region_attributes']
            category_name = region_attributes.get('label', default_category_name)
            category_id = category_dict.get(category_name)

            if category_id is None:
                warn_info_format = r'Warning: Ignore because category_name {} from image_file_name {} not in category_names {}'
                warn_info = warn_info_format.format(category_name,
                                                    image_file_name,
                                                    category_names)
                print(warn_info)
                continue

            is_crowd = 0

            shape_attributes = via_region['shape_attributes']
            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']

            points = [(x, y) for x, y in zip(all_points_x, all_points_y)]
            polygon = Polygon(points)

            bbox = list(polygon.bounds)

            segmentation = []

            for x, y in points:
                segmentation.extend([x, y])

            area = polygon.area

            coco_annotation = CocoFormat.create_annotation_info(annotation_id,
                                                                image_id,
                                                                category_id,
                                                                is_crowd,
                                                                bbox,
                                                                segmentation,
                                                                area)
            coco_annotations.append(coco_annotation)
            annotation_id += 1

        image_id += 1


    coco_output = {'images': coco_images,
                   'categories': coco_categories,
                   'annotations': coco_annotations}

    with open(Path(output_file_path).as_posix(), 'w') as file_stream:
        json.dump(coco_output, file_stream, indent=4)

    return coco_output


def main():
    image_dir_path = Path(r'D:/Data/cv/experiments/balloon_dataset/balloon/val')

    via_ann_file_path = image_dir_path / 'via_region_data.json'
    category_names = ['balloon']
    output_file_path = image_dir_path / 'test.json'

    convert(image_dir_path,
            via_ann_file_path,
            category_names,
            output_file_path=output_file_path,
            first_category_index=1)


if __name__ == '__main__':
    main()
