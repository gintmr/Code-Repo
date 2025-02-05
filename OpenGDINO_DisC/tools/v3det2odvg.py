import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO

def dump_label_map(args):
    # 加载COCO数据集
    coco = COCO(args.input) 
    # 加载所有类别
    cats = coco.loadCats(coco.getCatIds())
    # 将类别id和名称对应起来
    nms = {cat['id']-1:cat['name'] for cat in cats}
    # 将类别id和名称写入文件
    with open(args.output,"w") as f:
        json.dump(nms, f)

def coco_to_xyxy(bbox):
    # 将COCO格式的bbox转换为xyxy格式的bbox
    x, y, width, height = bbox
# 将变量x保留两位小数并赋值给变量x1
    x1 = round(x, 2) 
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]


def coco2odvg(args):
    # 加载COCO数据集
    coco = COCO(args.input) 
    # 加载所有类别
    cats = coco.loadCats(coco.getCatIds())
    # 将类别id和名称对应起来
    nms = {cat['id']:cat['name'] for cat in cats}
    metas = []
    # 遍历所有图片
    for img_id, img_info in tqdm(coco.imgs.items()):
        # 获取图片的所有标注id
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        # 遍历所有标注
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            # 获取标注的bbox
            bbox = ann['bbox']
            # 将bbox转换为xyxy格式
            bbox_xyxy = coco_to_xyxy(bbox)
            # 获取标注的类别id
            label = ann['category_id']
            # 获取类别名称
            category = nms[label]
            # 将标注信息添加到列表中
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": label - 1,       # make sure start from 0
                "category": category
                }
            )
        # 将图片信息添加到列表中
        metas.append(
            {
                "filename": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "detection": {
                    "instances": instance_list
                }
            }
        )
    print("  == dump meta ...")
    # 将图片信息写入文件
    with jsonlines.open(args.output, mode="w") as writer:
        writer.write_all(metas)
    print("  == done.")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("coco to odvg format.", add_help=True)
    # 添加输入文件参数
    parser.add_argument("--input", '-i', required=True, type=str, help="input list name")
    # 添加输出文件参数
    parser.add_argument("--output", '-o', required=True, type=str, help="output list name")
    # 添加输出标签映射参数
    parser.add_argument("--output_label_map", '-olm', action="store_true", help="output label map or not")
    # 解析命令行参数
    args = parser.parse_args()

    # 如果输出标签映射参数为真，则调用dump_label_map函数
    if args.output_label_map:
        dump_label_map(args)
    # 否则调用coco2odvg函数
    else:
        coco2odvg(args)