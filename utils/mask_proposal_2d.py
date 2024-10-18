import cv2
import supervision as sv
import os
import numpy as np
import cv2
import supervision as sv
import torch
import glob
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM
from utils.inference_florence import run_florence2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def process_views(num_views, screen_coords, pc_depth, save_dir, task_prompt, text_input, florence2_model, florence2_processor, sam2_predictor):
    output_dir = f"{save_dir}/sam_result/"
    os.makedirs(output_dir, exist_ok=True)
    visible_pts_list = []

    for view in range(num_views):
        screen_coor = screen_coords[view]  # 2D projected location of each 3D point
        point_idx = pc_depth[view]  # point index of each 2D pixel
        visible_pts = np.unique(point_idx)[1:]  # the first one is -1
        visible_pts_list.append(visible_pts)
        
        image_path = f"{save_dir}/rendered_img/{view}.png"
        image = Image.open(image_path).convert("RGB")
        img = cv2.imread(image_path)  # Load the image for annotation

        results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
        results = results[task_prompt]
        input_boxes = np.array(results["bboxes"])
        class_names = results["bboxes_labels"]
        class_ids = np.array(list(range(len(class_names))))

        # Predict mask with SAM 2
        sam2_predictor.set_image(np.array(image))
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Specify labels
        labels = [f"{class_name}" for class_name in class_names]

        # Visualize results
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator(color=sv.Color(r=255, g=0, b=0))
        annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
        
        output_image_path = os.path.join(output_dir, f"grounded_sam2_florence2_{view}.jpg")
        cv2.imwrite(output_image_path, annotated_frame)

    print('Done')
    return visible_pts_list



FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"








if __name__ == "__main__":

    save_dir = '/home/wan/Workplace-why/PartScene/output/test_pc'

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


    # build florence-2
    florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
    florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)


    # build sam 2
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    file_paths = glob.glob(os.path.join(img_dir, '*'))
    points_3d =[]
    visible_pts_list = []
    # Print all the files found

    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_input = "seat"


    process_views(9,save_dir,text_input,florence2_model,florence2_processor,sam2_predictor)