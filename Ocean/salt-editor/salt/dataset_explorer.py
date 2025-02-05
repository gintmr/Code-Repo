from pycocotools import mask
from skimage import measure 
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp
import os, cv2, copy
from distinctipy import distinctipy


def init_coco(dataset_folder, image_names, categories, coco_json_path):
    coco_json = {
        "info": {
            "description": "SAM Dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "Sam",
            "date_created": "2021/07/01",
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for i, category in enumerate(categories):
        coco_json["categories"].append(
            {"id": i, "name": category, "supercategory": category}
        )
    for i, image_name in enumerate(image_names):
        im = cv2.imread(os.path.join(dataset_folder, image_name))
        coco_json["images"].append(
            {
                "id": i,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
    with open(coco_json_path, "w") as f:
        json.dump(coco_json, f)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def parse_mask_to_coco(image_id, anno_id, image_mask, category_id, poly=False):
    start_anno_id = anno_id
    x, y, width, height = bounding_box_from_mask(image_mask)
    if poly == False:
        fortran_binary_mask = np.asfortranarray(image_mask)
        encoded_mask = mask.encode(fortran_binary_mask)
    if poly == True:
        contours = measure.find_contours(image_mask, 0.5)
    annotation = {
        "id": start_anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [],
    }
    if poly == False:
        annotation["segmentation"] = encoded_mask
        annotation["segmentation"]["counts"] = str(
            annotation["segmentation"]["counts"], "utf-8"
        )
    if poly == True:
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            sc = bunch_coords(segmentation)
            sc = simplify_coords_vwp(sc, 2)
            sc = unbunch_coords(sc)
            annotation["segmentation"].append(sc)
    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, categories=None, coco_json_path=None):
        # 初始化数据集探索器
        self.dataset_folder = dataset_folder
        # 获取图像文件夹中的所有图像文件名
        self.image_names = sorted(os.listdir(os.path.join(self.dataset_folder, "images")))
        # 过滤出以.jpg、.png、.jpeg、.JPG结尾的文件名
        self.image_names = [
            os.path.split(name)[1]
            for name in self.image_names
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg") or name.endswith(".JPG")
        ]
        self.coco_json_path = coco_json_path
        # 如果coco_json_path不存在，则初始化coco_json
        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)
        # 读取coco_json文件
        with open(coco_json_path, "r") as f:
            self.coco_json = json.load(f)

        # 获取coco_json中的所有类别
        self.categories = [
            category["name"] for category in self.coco_json["categories"]
        ]
        # 获取每个图像的注释
        self.annotations_by_image_id = {}
        for annotation in self.coco_json["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(annotation)

        # 初始化全局注释id
        self.global_annotation_id = len(self.coco_json["annotations"])
        # 获取每个类别的颜色
        self.category_colors = distinctipy.get_colors(len(self.categories))
        self.category_colors = [
            tuple([int(255 * c) for c in color]) for color in self.category_colors
        ]

    def __init_coco_json(self, categories):
        # 初始化coco_json
        appended_image_names = [
            os.path.join("images", name) for name in self.image_names
        ]
        # 将self.image_names中的每个name与"images"拼接，生成新的appended_image_names列表
        init_coco(
            self.dataset_folder, appended_image_names, categories, self.coco_json_path
        )

    def get_colors(self, category_id):
        # 获取指定类别的颜色
        return self.category_colors[category_id]

    def get_last_imageid(self,annotations):
        # 获取最后一个注释的图像id
        last_anno=annotations[-1]
        image_id=last_anno['image_id']
        return image_id

    def get_last_annoid(self,annotations):
        # 获取最后一个注释的id
        last_anno=annotations[-1]
        id=last_anno['id']
        return id

    def get_categories(self, get_colors=False):
        # 获取所有类别
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def get_num_images(self):
        # 获取图像数量
        return len(self.image_names)

    def get_image_data(self, image_id):
        # 获取图像数据
        image_name = self.coco_json["images"][image_id]["file_name"]
        image_path = os.path.join(self.dataset_folder, image_name)
        embedding_path = os.path.join(
            self.dataset_folder,
            "embeddings",
            os.path.splitext(os.path.split(image_name)[1])[0] + ".npy",
        )
        image = cv2.imread(image_path)
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_embedding = np.load(embedding_path)
        return image, image_bgr, image_embedding,os.path.basename(image_name)

    def __add_to_our_annotation_dict(self, annotation):
        # 将注释添加到我们的注释字典中
        image_id = annotation["image_id"]
        if image_id not in self.annotations_by_image_id:
            self.annotations_by_image_id[image_id] = []
        self.annotations_by_image_id[image_id].append(annotation)

    def get_annotations(self, image_id, return_colors=False):
        # 获取指定图像的注释
        if image_id not in self.annotations_by_image_id:
            return [], []
        cats = [a["category_id"] for a in self.annotations_by_image_id[image_id]]
        colors = [self.category_colors[c] for c in cats]
        if return_colors:
            return self.annotations_by_image_id[image_id], colors
        return self.annotations_by_image_id[image_id]

    def update_annotations(self, image_id, annotation_ids, new_label):
        # print("Update label to", new_label, "for", annotation_ids, "in dataexplorer")
        label_id = self.get_categories().index(new_label)

        for annotation_id in annotation_ids:
            for annotation in self.coco_json["annotations"]:
                if (
                    annotation["image_id"] == image_id
                    and annotation["id"] == annotation_id
                ):
                    annotation["category_id"] = label_id

        for annotation in self.annotations_by_image_id[image_id]:
            if annotation["id"] in annotation_ids:
                annotation["category_id"] = label_id

    def clear_annotations(self, image_id, annotation_ids):
        # 遍历annotation_ids中的每个annotation_id
        for annotation_id in annotation_ids:
            # 遍历coco_json中的每个annotation
            for annotation in self.coco_json["annotations"]:
                # 如果annotation的image_id和annotation_id都匹配
                if (
                    annotation["image_id"] == image_id
                    and annotation["id"] == annotation_id
                ):  # and annotation["id"] in annotation_ids:
                    # 从coco_json中删除匹配的annotation
                    self.coco_json["annotations"].remove(annotation)

        # iterate over a copy of the list annotaiton_by_image_id[image_id]
        # because implace modification of the list causes discrapancies with the list index
        for annotation in self.annotations_by_image_id[image_id][:]:
            if annotation["id"] in annotation_ids:
                self.annotations_by_image_id[image_id].remove(annotation)

    def add_annotation(self, image_id, category_id, mask, poly=True):
        # 将poly参数设置为False
        poly=False
        # 如果mask为空，则返回
        if mask is None:
            return
        # 将mask转换为coco格式
        annotation = parse_mask_to_coco(
            image_id, self.global_annotation_id, mask, category_id, poly=poly
        )
        # 将annotation添加到我们的注释字典中
        self.__add_to_our_annotation_dict(annotation)
        # 将annotation添加到coco_json的annotations列表中
        self.coco_json["annotations"].append(annotation)
        # 全局注释id加1
        self.global_annotation_id += 1

    def save_annotation(self):
        with open(self.coco_json_path, "w") as f:
            json.dump(self.coco_json, f)
