import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
from groundingdino.datasets import transforms as T
from models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span



def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ## image_pil [1920, 1280]
    ## image, tensor [800*1200]
    
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    # 从配置文件中加载模型参数
    args = SLConfig.fromfile(model_config_path)
    # 如果不使用cpu_only，则使用cuda，否则使用cpu
    args.device = "cuda" if not cpu_only else "cpu"
    # 根据参数构建模型
    model, criterion_DisC, postprocessors = build_model(args)
    # 加载模型参数
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    # 加载模型参数到模型中，不严格匹配
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # 打印加载结果
    print(load_res)
    # 将模型设置为评估模式
    _ = model.eval()
    # 返回模型
    return model


from models.GroundingDINO.utils import ContrastiveEmbed

def get_grounding_output(DisC_model, model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    # device = "cuda" if not cpu_only else "cpu"
    device = 'cuda'
    model = model.to(device)
    image = image.to(device)
    # 在不计算梯度的情况下，使用模型对图像和描述进行预浄1�7
    with torch.no_grad():
        outputs, filt_outputs, pred_phrases, text_dict, filt_boxes = model(image[None], captions=[caption])
        filt_output_reshaped = np.concatenate([output.to('cpu').detach().numpy() for output in filt_outputs], axis=0)           
        filt_output_reshaped = torch.from_numpy(filt_output_reshaped)
        filt_faeture = filt_output_reshaped.to(device)
        DisC_model = DisC_model.to(device)
        # filt_faeture = DisC_model(filt_faeture)
        res = filt_faeture @ text_dict['encoded_text'].transpose(-1, -2)
        # 将text_dict中的'text_token_mask'取反，然后与float("-inf")相乘，最后将结果填充到res丄1�7
        res.masked_fill_(~text_dict['text_token_mask'][:, None, :], float("-inf"))
        new_res = torch.full((*res.shape[:-1], 256), float("-inf"), device=res.device)
        new_res[..., : res.shape[-1]] = res
        out = new_res
        
        
        
        
    
    # logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)  ##这里取[0]代表bs的第丢�个，就是默认的inference on one picture。此时输出的256就已经是对应的1�7256个输入点位了
    # boxes = outputs["pred_boxes"][0]  # (nq, 4)
    boxes = torch.tensor(filt_boxes[0])
    logits = out.squeeze(0) ##squeese将logits的维度从(1＄1�71＄1�7256)变为(1, 256)
    logits = torch.softmax(logits, dim=1) ## 对logits的第二个维度进行softmax操作，得到概率分帄1�7


    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        # 获取logits_filt中每丢�行的朢�大��，并返回最大��和对应的索弄1�7
        logits_filt_max = logits_filt.max(dim=1)[0]
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold ## 丢�个维度为[900]的布尔型张量，表示每个位置的logits是否大于box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        
        
        
        # pred_phrases = []
        # for logit, box in zip(logits_filt, boxes_filt):
        #     pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        #     if with_logits:
        #         pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        #     else:
        #         pred_phrases.append(pred_phrase)
        
        
        
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256
        ## positive_maps[2, 256], logits[900, 256]
        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq ## 丢�个正映射（positive map），该映射表示输入文本中的短语与模型输出中的查询之间的关系��具体来说，如果输入文本中的短语与模型输出中的查询在同一个位置，那么对应的映射��为 True，否则为 False〄1�7
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default = '/data2/wuxinrui/Open-GroundingDino/config/cfg_coco.py' , help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default='/data2/wuxinrui/OpenGDINO_DisC/weights/groundingdino_swint_ogc.pth',  help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, default='/data2/wuxinrui/Open-GroundingDino/test/33.jpg', help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, default= 'horse' , help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs",   help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    # parser.add_argument("--token_spans", type=str, default=[[[2, 6]],[[14, 19]]], help=
    # parser.add_argument("--token_spans", type=str, default=[[[0, 5]]], help=
    parser.add_argument("--token_spans", type=str, default=None, help=

                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")


    from models.GroundingDINO.DisCutils.utils import Adjust_Module


    DisC_model = Adjust_Module()
    weights = torch.load('/data2/wuxinrui/OpenGDINO_DisC/outputs/checkpoint0029.pth', map_location='cpu')
    DisC_model.load_state_dict(weights['DisC_model'])

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        DisC_model, model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=token_spans
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    save_path = os.path.join(output_dir, "pred.jpg")
    image_with_box.save(save_path)
    print(f"\n======================\n{save_path} saved.\nThe program runs successfully!")
