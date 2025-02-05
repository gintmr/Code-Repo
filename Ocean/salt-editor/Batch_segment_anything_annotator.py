import os
import argparse
import sys
import logging
import shutil

# 导入PyQt5中的QApplication模块
from PyQt5.QtWidgets import QApplication

from salt.editor import Editor
from salt.interface import ApplicationInterface


def get_valid_datasets(task_path):
    """
    获取 task_path 下所有有效的子文件夹（包含 models、images 和 annotations.json）。
    """
    valid_datasets = []
    for folder_name in os.listdir(task_path):
        folder_path = os.path.join(task_path, folder_name)
        if os.path.isdir(folder_path):  # 确保是文件夹
            models_path = os.path.join(folder_path, "models")
            images_path = os.path.join(folder_path, "images")
            annotations_path = os.path.join(folder_path, "annotations.json")
            
            # 检查是否包含必要的文件和文件夹
            if (os.path.exists(models_path) and
                os.path.exists(images_path) and
                os.path.exists(annotations_path)):
                valid_datasets.append(folder_path)
    return valid_datasets


class DatasetAnnotator:
    def __init__(self, task_path, categories, finished_path):
        self.task_path = task_path
        self.categories = categories
        self.finished_path = finished_path
        self.valid_datasets = get_valid_datasets(task_path)
        self.current_dataset_index = 0

        # 初始化日志
        self.setup_logging()

        if not self.valid_datasets:
            logging.error("No valid datasets found in the task path.")
            sys.exit(1)

        # 初始化应用程序
        self.app = QApplication(sys.argv)

        # 加载第一个数据集
        self.load_dataset()

    def setup_logging(self):
        """
        设置日志功能，确保日志不会被覆盖。
        """
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别为INFO
            format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
            handlers=[
                logging.FileHandler("dataset_annotator.log", mode='a'),  # 使用追加模式，将日志写入文件
                logging.StreamHandler()  # 将日志输出到控制台
            ]
        )

    def load_dataset(self):
        """
        加载当前索引指向的数据集。
        """
        # 获取当前数据集的路径
        current_dataset_path = self.valid_datasets[self.current_dataset_index]
        # 获取当前数据集的模型路径
        onnx_models_path = os.path.join(current_dataset_path, "models")
        # 获取当前数据集的图片路径
        images_path = os.path.join(current_dataset_path, "images")
        # 获取当前数据集的图片数量
        images_len = len(os.listdir(images_path))
        # 获取当前数据集的COCO注释文件路径
        coco_json_path = os.path.join(current_dataset_path, "annotations.json")

        # 创建Editor对象
        self.editor = Editor(
            onnx_models_path,
            current_dataset_path,
            categories=self.categories,
            coco_json_path=coco_json_path
        )

        # 创建窗口
        self.window = ApplicationInterface(self.app, self.editor, list_num=images_len)

        # 绑定关闭事件以加载下一个数据集
        self.window.closeEvent = self.on_close_event

        # 显示窗口
        self.window.show()

        # 记录当前处理的文件夹
        # 记录当前处理的文件夹
        remaining_datasets = len(self.valid_datasets) - self.current_dataset_index - 1
        logging.info(f"Processing dataset: {current_dataset_path}")
        logging.info(f"Remaining datasets in parent-folder({task_path}): {remaining_datasets}")

    def on_close_event(self, event):
        """
        关闭事件处理函数，加载下一个数据集，并将当前数据集移动到 finished 文件夹。
        """
        current_dataset_path = self.valid_datasets[self.current_dataset_index]
        self.move_to_finished(current_dataset_path)

        self.current_dataset_index += 1
        if self.current_dataset_index < len(self.valid_datasets):
            # 加载下一个数据集
            self.load_dataset()
        else:
            # 所有数据集已处理完毕
            logging.info("All datasets have been processed.")
            self.app.quit()  # 退出应用程序

        event.accept()  # 接受关闭事件

    def move_to_finished(self, dataset_path):
        """
        将处理完的数据集移动到 finished 文件夹。
        """
        # 检查 finished 文件夹是否存在，如果不存在则创建
        if not os.path.exists(self.finished_path):
            os.makedirs(self.finished_path)

        # 获取数据集的名称
        dataset_name = os.path.basename(dataset_path)
        # 构建目标路径
        destination_path = os.path.join(self.finished_path, dataset_name)

        # 尝试移动数据集到目标路径
        try:
            shutil.move(dataset_path, destination_path)
            # 记录日志
            logging.info(f"Moved dataset {dataset_name} to {destination_path}")
        except Exception as e:
            # 记录错误日志
            logging.error(f"Failed to move dataset {dataset_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=str, default="D:\\Ocean\\ICCV\\finished")
    parser.add_argument("--categories", type=str, default="massive,encrusting,branching,foliaceous,columnar,laminar,free,soft,sponge")
    parser.add_argument("--finished-path", type=str, default="D:\\Ocean\\ICCV\\@batch_4")
    args = parser.parse_args()

    task_path = args.task_path
    categories = args.categories.split(",") if args.categories else None
    finished_path = args.finished_path

    # 启动标注器
    annotator = DatasetAnnotator(task_path, categories, finished_path)
    sys.exit(annotator.app.exec_())  # 启动事件循环