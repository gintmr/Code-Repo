import cv2
import numpy as np
from pycocotools import mask as coco_mask


class DisplayUtils:
    def __init__(self):
        self.transparency = 0.6
        self.box_width = 2
        self.hover_transparency = 0.8
        self.hover_mask_id = []

    def increase_transparency(self):
        self.transparency = min(1.0, self.transparency + 0.05)

    def decrease_transparency(self):
        self.transparency = max(0.0, self.transparency - 0.05)

    def overlay_mask_on_image(self, image, mask, color=(255, 0, 0), hover=False):
        gray_mask = mask.astype(np.uint8) * 255
        gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
        color_mask = cv2.bitwise_and(gray_mask, color)
        masked_image = cv2.bitwise_and(image.copy(), color_mask)
        if hover:
            # print("hover transparancy", hover)
            overlay_on_masked_image = cv2.addWeighted(
                masked_image, self.hover_transparency, color_mask, 1 - self.hover_transparency, 0
            )
        else:
            # print("normal transparancy", hover)
            overlay_on_masked_image = cv2.addWeighted(
                masked_image, self.transparency, color_mask, 1 - self.transparency, 0
            )
        background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
        image = cv2.add(background, overlay_on_masked_image)
        return image

    def __convert_ann_to_mask(self, ann, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)

        rle = ann["segmentation"]

        # polygon
        # poly = ann["segmentation"]
        # used_poly=[]
        # for tmp in poly:
        #     if len(tmp)>8:
        #         used_poly.append(tmp)
        # rles = coco_mask.frPyObjects(used_poly, height, width)
        # rle = coco_mask.merge(rles)


        mask_instance = coco_mask.decode(rle)
        mask_instance = np.logical_not(mask_instance)
        mask = np.logical_or(mask, mask_instance)
        mask = np.logical_not(mask)
        return mask

    def masks_containing_pts(self, pt, image, annotations, colors):
        mask_ids = []
        for ann, color in zip(annotations, colors):
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            # print(mask[int(pt[1]), int(pt[0])])
            if mask[int(pt[1]), int(pt[0])]:
                mask_ids.append(ann['id'])
        return mask_ids

    def draw_box_on_image(self, image, ann, color):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if color == (0, 0, 0):
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        else:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)
        image = cv2.putText(
            image,
            "id: " + str(ann["id"]),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        return image

    def draw_annotations(self, image, annotations, colors):
        for ann, color in zip(annotations, colors):
            image = self.draw_box_on_image(image, ann, color)
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            if ann["id"] in self.hover_mask_id:
                # print(ann["id"], "hover true")
                image = self.overlay_mask_on_image(image, mask, color, hover=True)
            else:
                # print(ann["id"], "hover false")
                image = self.overlay_mask_on_image(image, mask, color, hover=False)
        return image

    def draw_points(
        self, image, points, labels, colors={1: (0, 255, 0), 0: (0, 0, 255)}, radius=5
    ):
        for i in range(points.shape[0]):
            point = points[i, :]
            label = labels[i]
            color = colors[label]
            image = cv2.circle(image, tuple(point), radius, color, -1)
        return image
