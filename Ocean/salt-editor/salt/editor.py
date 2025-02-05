import os, copy
import numpy as np
from salt.onnx_model import OnnxModels
from salt.dataset_explorer import DatasetExplorer
from salt.display_utils import DisplayUtils

selected_annotations = []


class CurrentCapturedInputs:
    def __init__(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def reset_inputs(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        self.curr_mask = mask

    def add_input_click(self, input_point, input_label):
        if len(self.input_point) == 0:
            self.input_point = np.array([input_point])
        else:
            self.input_point = np.vstack([self.input_point, np.array([input_point])])
        self.input_label = np.append(self.input_label, input_label)

    def set_low_res_logits(self, low_res_logits):
        self.low_res_logits = low_res_logits


class Editor:
    def __init__(
        self, onnx_models_path, dataset_path, categories=None, coco_json_path=None
    ):
        # 初始化函数，传入onnx模型路径、数据集路径、类别和coco_json路径
        self.dataset_path = dataset_path
        # 数据集路径
        self.coco_json_path = coco_json_path
        # coco_json路径
        if categories is None and not os.path.exists(coco_json_path):
            # 如果类别为空且coco_json路径不存在，则抛出异常
            raise ValueError("categories must be provided if coco_json_path is None")
        if self.coco_json_path is None:
            # 如果coco_json路径为空，则将coco_json路径设置为数据集路径下的annotations.json
            self.coco_json_path = os.path.join(self.dataset_path, "annotations.json")
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path, categories=categories, coco_json_path=self.coco_json_path
        )
        # 创建数据集探索器
        self.curr_inputs = CurrentCapturedInputs()
        # 创建当前捕获的输入
        self.categories, self.category_colors = self.dataset_explorer.get_categories(
            get_colors=True
        )
        # 获取类别和类别颜色
        self.image_id = 0
        self.category_id = 0
        self.show_other_anns = True
        # 初始化图像id、类别id和是否显示其他注释
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        # 获取图像数据
        if self.image_name.endswith(".jpeg"):
            # 如果图像名称以.jpeg结尾，则去掉.jpeg
            self.image_name=self.image_name[:-5]
        else:
            # 否则去掉.jpg
            self.image_name = self.image_name[:-4]
        self.display = self.image_bgr.copy()
        # 复制图像数据
        self.onnx_helper = OnnxModels(
            onnx_models_path,
            image_width=self.image.shape[1],
            image_height=self.image.shape[0],
            name = self.image_name,
        )
        # 创建onnx模型助手
        self.du = DisplayUtils()
        self.reset()

    def list_annotations(self):
        # 获取图像的注释和颜色
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        # 返回注释和颜色
        return anns, colors

    def update_annotation_label(self, new_label, annotation_ids):
        self.dataset_explorer.update_annotations(self.image_id, annotation_ids, new_label)

    def clear_annotations(self, annotation_ids):
        self.dataset_explorer.clear_annotations(self.image_id, annotation_ids)

    def get_last_imageid(self):
        return self.dataset_explorer.get_last_imageid(self.dataset_explorer.coco_json["annotations"])

    def get_last_annoid(self):
        return self.dataset_explorer.get_last_annoid(self.dataset_explorer.coco_json["annotations"])

    def __draw_known_annotations(self, selected_annotations=[]):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        for i, (ann, color) in enumerate(zip(anns, colors)):
            for selected_ann in selected_annotations:
                if ann["id"] == selected_ann:
                    colors[i] = (0, 0, 255)
        # Use this to list the annotations
        self.display = self.du.draw_annotations(self.display, anns, colors)

    def __draw(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        if self.curr_inputs.curr_mask is not None:
            self.display = self.du.draw_points(
                self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
            )
            self.display = self.du.overlay_mask_on_image(
                self.display, self.curr_inputs.curr_mask
            )
        if self.show_other_anns:
            self.__draw_known_annotations(selected_annotations)

    def add_click(self, new_pt, new_label, selected_annotations=[]):
        # 添加一个新的点击
        self.curr_inputs.add_input_click(new_pt, new_label)
        # 调用onnx_helper.call方法，传入当前图像、图像嵌入、输入点、输入标签和低分辨率对数
        masks, low_res_logits = self.onnx_helper.call(
            self.image,
            self.image_embedding,
            self.curr_inputs.input_point,
            self.curr_inputs.input_label,
            low_res_logits=self.curr_inputs.low_res_logits,
        )
        # 设置当前输入的掩码
        self.curr_inputs.set_mask(masks[0, 0, :, :])
        # 设置当前输入的低分辨率对数
        self.curr_inputs.set_low_res_logits(low_res_logits)
        # 绘制选定的注释
        self.__draw(selected_annotations)

    def remove_click(self, new_pt):
        # 打印运行remove click
        print("ran remove click")
        # 获取new_pt的mask_id
        remove_mask_ids = self.get_pt_mask(new_pt)
        # 清除remove_mask_ids中的注释
        self.clear_annotations(remove_mask_ids)
        # 绘制selected_annotations
        # self.__draw(selected_annotations)

    # 根据给定的点，选择对应的注释
    def choose_annotation(self, point):
        # 获取给定点的掩码
        return self.get_pt_mask(point)

    # 重置函数，hard参数表示是否完全重置，selected_annotations参数表示选中的注释
    def reset(self, hard=True, selected_annotations=[]):
        # 重置当前输入
        self.curr_inputs.reset_inputs()
        # 绘制选中的注释
        self.__draw(selected_annotations)

    # 定义一个toggle函数，用于切换显示其他标注
    def toggle(self, selected_annotations=[]):
        # 将show_other_anns属性取反
        self.show_other_anns = not self.show_other_anns
        # 调用__draw函数，传入selected_annotations参数
        self.__draw(selected_annotations)

    def get_pt_mask(self, pt):
        masks = []
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        image = self.display
        masks_ids = self.du.masks_containing_pts(pt, image, anns, colors)
        return masks_ids
        # print(mask[pt])

    def hover(self, pt=[], selected_annotations=[]):
        # 获取鼠标悬停位置的掩膜ID
        self.du.hover_mask_id = self.get_pt_mask(pt)
        # 复制图像
        self.display = self.image_bgr.copy()
        # 绘制已知的注释
        self.__draw_known_annotations(selected_annotations)
        # 清空鼠标悬停位置的掩膜ID
        self.du.hover_mask_id = []

    # 定义一个函数，用于增加透明度
    def step_up_transparency(self, selected_annotations=[]):
        # 复制图像
        self.display = self.image_bgr.copy()
        # 增加透明度
        self.du.increase_transparency()
        # 绘制选中的注释
        self.__draw(selected_annotations)

    # 减少透明度
    def step_down_transparency(self, selected_annotations=[]):
        # 复制图像
        self.display = self.image_bgr.copy()
        # 减少透明度
        self.du.decrease_transparency()
        # 绘制选中的注释
        self.__draw(selected_annotations)

    # 绘制选中的注释
    def draw_selected_annotations(self, selected_annotations=[]):
        # 调用__draw方法，传入选中的注释
        self.__draw(selected_annotations)

    def save_ann(self):
        # 调用dataset_explorer对象的add_annotation方法，传入image_id，category_id和curr_inputs.curr_mask
        self.dataset_explorer.add_annotation(
            self.image_id, self.category_id, self.curr_inputs.curr_mask
        )

    def save(self):
        self.dataset_explorer.save_annotation()

    def next_image(self):
        # 如果当前图片id等于数据集探索器中的图片数量减1，则返回
        if self.image_id == self.dataset_explorer.get_num_images() - 1:
            return
        # 当前图片id加1
        self.image_id += 1
        # 获取当前图片的数据
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        # 复制当前图片的bgr数据
        self.display = self.image_bgr.copy()
        # 如果图片名称以.jpeg结尾，则去掉.jpeg
        if self.image_name.endswith(".jpeg"):
            self.image_name=self.image_name[:-5]
        # 否则去掉.jpg
        else:
            self.image_name = self.image_name[:-4]
        # 设置onnx助手中的图片分辨率
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0],name = self.image_name)
        # 重置
        self.reset()

    def set_image_by_imageid(self,image_id,anno_id):
        if image_id == self.dataset_explorer.get_num_images() - 1:
            return
        self.image_id = image_id
        self.dataset_explorer.global_annotation_id=anno_id+1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        if self.image_name.endswith(".jpeg"):
            self.image_name=self.image_name[:-5]
        else:
            self.image_name = self.image_name[:-4]
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0],name = self.image_name)
        self.reset()

    def prev_image(self):
        if self.image_id == 0:
            return
        self.image_id -= 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        if self.image_name.endswith(".jpeg"):
            self.image_name=self.image_name[:-5]
        else:
            self.image_name = self.image_name[:-4]
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0],name = self.image_name)
        self.reset()

    def next_category(self):
        if self.category_id == len(self.categories) - 1:
            self.category_id = 0
            return
        self.category_id += 1

    def prev_category(self):
        if self.category_id == 0:
            self.category_id = len(self.categories) - 1
            return
        self.category_id -= 1

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def select_category(self, category_name):
        category_id = self.categories.index(category_name)
        self.category_id = category_id
