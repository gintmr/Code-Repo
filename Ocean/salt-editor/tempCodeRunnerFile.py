import os
import argparse
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QRadioButton, QAbstractItemView
from PyQt5.QtCore import QTimer, QTime, Qt

from salt.editor import Editor
from salt.interface import ApplicationInterface

class AutoNextApplicationInterface(ApplicationInterface):
    def __init__(self, app, editor, list_num, paths, task_path):
        super().__init__(app, editor, list_num)
        self.paths = paths
        self.task_path = task_path
        self.current_index = 0

    def closeEvent(self, event):
        # 关闭当前窗口时，加载下一个标注框
        self.current_index += 1
        if self.current_index < len(self.paths):
            self.load_next_dataset()
            self.show()  # 显示新窗口
        event.accept()

    def load_next_dataset(self):
        path = self.paths[self.current_index]
        dataset_path = os.path.join(self.task_path, path)
        
        onnx_models_path = os.path.join(dataset_path, "models")
        images_path = os.path.join(dataset_path, "images")
        images_len = len(os.listdir(images_path))
        
        categories = None
        if args.categories is not None:
            categories = args.categories.split(",")
        
        coco_json_path = os.path.join(dataset_path, "annotations.json")

        editor = Editor(
            onnx_models_path,
            dataset_path,
            categories=categories,
            coco_json_path=coco_json_path
        )

        self.editor = editor
        self.list_num = images_len
        self.update_interface()

    def update_interface(self):
        # 清除旧界面并设置新界面
        self.clear_interface()
        self.setup_interface()  # 调用父类的 setup_interface 方法

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=str, default="D:\\Ocean\\ICCV\\test")
    parser.add_argument("--categories", type=str, default="massive,encrusting,branching,foliaceous,columnar,laminar,free,soft,sponge")
    args = parser.parse_args()

    task_path = args.task_path
    paths = os.listdir(task_path)

    if paths:
        dataset_path = os.path.join(task_path, paths[0])
        
        onnx_models_path = os.path.join(dataset_path, "models")
        images_path = os.path.join(dataset_path, "images")
        images_len = len(os.listdir(images_path))
        
        categories = None
        if args.categories is not None:
            categories = args.categories.split(",")
        
        coco_json_path = os.path.join(dataset_path, "annotations.json")

        editor = Editor(
            onnx_models_path,
            dataset_path,
            categories=categories,
            coco_json_path=coco_json_path
        )

        app = QApplication(sys.argv)
        window = AutoNextApplicationInterface(app, editor, images_len, paths, task_path)
        window.show()
        sys.exit(app.exec_())