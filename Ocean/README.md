## 核心关注
- Ocean\salt-editor\Batch_segment_anything_annotator.py ：经过~~漫长~~的修改过程，终于为segmenta_anything模块加上批处理功能。
    
    功能详解：

    1. 遍历标注父文件夹中所有的子文件夹
    2. 上一个文件夹标注完成后，自动将文件夹移动至FINISHED中，并且在打印台输出当前剩余文件夹个数
- 