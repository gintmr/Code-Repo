import os
import argparse
import sys

from PyQt5.QtWidgets import QApplication

from salt.editor import Editor
from salt.interface import ApplicationInterface
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--onnx-models-path", type=str, default="./models")
    # parser.add_argument("--dataset-path", type=str, default="./dataset")
    # parser.add_argument("--task-path", type=str, default="D:\Ocean\@Select\\new_select")
    # parser.add_argument("--task-path", type=str, default="D:/Ocean/zzq_long6_first")
    parser.add_argument("--task-path", type=str, default="D:\\Ocean\\ICCV\\test\\Alcithoe_arabica")
    


    parser.add_argument("--categories", type=str,default="massive,encrusting,branching,foliaceous,columnar,laminar,free,soft,sponge")
    args = parser.parse_args()

    task_path = args.task_path
    
    dataset_path = os.path.join(task_path)
    
    onnx_models_path = os.path.join(task_path, "models")
    images_path = os.path.join(task_path, "images")
    images_len = len(os.listdir(images_path))
    
    categories = None
    if args.categories is not None:
        categories = args.categories.split(",")
    
    coco_json_path = os.path.join(task_path,"annotations.json")

    editor = Editor(
        onnx_models_path,
        dataset_path,
        categories=categories,
        coco_json_path=coco_json_path
    )

    app = QApplication(sys.argv)
    window = ApplicationInterface(app, editor, list_num = images_len)
    window.show()
    sys.exit(app.exec_())
