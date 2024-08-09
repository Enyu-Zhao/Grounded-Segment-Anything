import os, sys, shutil
import json
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO


import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,BitsAndBytesConfig
import openai as OpenAI
# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import re
import random
from huggingface_hub import hf_hub_download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import extract_number,get_sorted_files,numpy_to_base64, add_grid_to_image, encode_image


DEVICE_0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DEVICE_1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')




def count_objects(sentence):
    # Use regular expression to find all occurrences of the pattern ending with "."
    matches = re.findall(r'[^.]+\s*\.', sentence)
    return len(matches)


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model 




# Load Grounding DINO
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, DEVICE_0)


# Load SAM
sam_checkpoint = './sam_weight.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(DEVICE_0))



with open("../GPT-API-Key.txt", "r") as f:
    api_key = f.read().strip()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def get_result_from_VLM(model,tokenizer,messages):

    inputs = tokenizer.apply_chat_template(messages,
                                    add_generation_prompt=True, 
                                    tokenize=True, 
                                    return_tensors="pt",
                                    return_dict=True)
    inputs = inputs.to(DEVICE_0)
    gen_kwargs = {"max_length": 2500, "do_sample": False, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response=tokenizer.decode(outputs[0])

    


    return response[:-13]


def communicate_gpt(messages,headers=headers):
    """
    get response from GPT-4 API. Content is the input to the API.
    
    """
    
    payload={
            "model":"gpt-4o",
            "messages":messages,
            "max_tokens": 1024,
            "temperature":0,
            "top_p":1,
            "frequency_penalty":0,
            "presence_penalty":0
        }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    
    if 'choices' in response.json():
        # GPT doesn't run into error.                
        response_message=response.json()['choices'][0]['message']['content']
    else:
        print(response.json())
        response_message=None   
        
    return response_message


def VLM_guided_detect(image_tensor, image_np, text_prompt, VLM_model, VLM_tokenizer,cv_model=groundingdino_model, box_threshold = 0.2, text_threshold = 0.2, system_prompt=None, user_prompt=None): 

    if system_prompt is None:
        system_prompt="""
            Role: You will provide some guidance on which object to segment for vision models.
            Task: You will be given a natural language description about the task and an image.Now based on the description and the image, you will provide the objects that you think are important to completing the task.
            Output Requirement: Only output the objects description that you think are important to completing the task. For each object, separate them with a space and a period.
            
            I am giving you an example now.
            
            Example 1:
            Task Description: pick up the yellow cup.
            Output: yellow cup . 
            
            
            Example 2:
            Task Description: Move the silver vessel and place it below the spoon.
            Output: silver vessel . spoon .
            

            
            
        """
    
    
    
    another_example="""        
        Example 3:
        Task Description: Build a tool hang by first picking up the L-shaped pole and then piercing it through the hole in the wooden stand, then hang the tool to the tip of the L-shaped pole.
        Output: L-shaped pole. hole. wooden stand. tool.
        """
    
    
    if user_prompt is None:  
        user_prompt=f"""
        Now, with the given task description and the image, please provide the objects that you think are important to completing the task.
        
        Task Description: {text_prompt}.
        """
    
    
    if VLM_model is None:
        print("using GPT-4o now.")
        
        messages=[{
            "role":"system",
            "content":system_prompt,
        }]
        messages.append({
        "role":"user",
        "content":[user_prompt,{"image":numpy_to_base64(image_np)}],
        })
        
        response=communicate_gpt(messages,headers=headers)
        
    else:   
        messages=[{
            "role":"system",
            "content":system_prompt,
            # "image":Image.open("./VLM_example_image.jpg")
        }]
        

        messages.append({
            "role":"user",
            "content":user_prompt,
            "image":Image.fromarray(image_np)
        })
        response= get_result_from_VLM(VLM_model,VLM_tokenizer,messages)
        
        
    print(f"previous description: {text_prompt}")
    print(f"trimmed down version:{response}")
    annotated_frames, boxes=detect(image_tensor, image_np, response, cv_model, box_threshold, text_threshold)
    
    return annotated_frames,boxes,response



def get_objects_from_text(image_tensor, image_np,text_prompt,system_prompt=None,user_prompt=None):
    # In this function, we will use GPT-4 to get the objects from the text description
    
    
    if system_prompt is None:
        system_prompt="""
            Role: You will provide some guidance on which object to segment for vision models.
            Task: You will be given a natural language description about the task and an image. Now based on the description and the image, you will provide the object that you think the robot should grasp.
            Output Requirement: Only output the one object that the robot should grasp to complete the given task. 
            
            I am giving you examples now.
            
            Example 1:
            Task Description: pick up the yellow cup.
            Output: yellow cup . 
            
            
            Example 2:
            Task Description: Close the microwave.
            Output: microwave door handle .
            
            Example 3:
            Task Description: Open the left cabinet door.
            Output: left cabinet door .  
            
        """
        
    if user_prompt is None:
        user_prompt=f"""
        Now, with the given task description and the image, please provide the object that you think the robot should grasp to complete the task.
        
        Task Description: {text_prompt}.
        """
        
    
    messages=[{
            "role":"system",
            "content":system_prompt,
        }]
    messages.append({
    "role":"user",
    "content":[user_prompt,{"image":numpy_to_base64(image_np)}],
    })
    
    response=communicate_gpt(messages,headers=headers)
        
    return response




def detect(image_tensor, image_np, text_prompt, model, box_threshold = 0.2, text_threshold = 0.2):
  boxes, logits, phrases = predict(
      model=model, 
      image=image_tensor, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
  return annotated_frame, boxes 


def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(DEVICE_0), image.shape[:2])

  masks, A, B = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  
  return masks.cpu()
  
##########Reference Functions from GroundedSAM##########
def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def draw_mask_contour(mask, image, random_color=True):
    # Convert mask to a binary image
    image = image.astype(np.uint8)
    mask = mask.numpy()
    # _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    binary_mask = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Decide on the color
    if random_color:
        color = np.random.random(3)
    else:
        color = [30, 144, 255]  # Deep sky blue in BGR format
    # Draw the contours
    cv2.drawContours(image, contours, -1, color, thickness=2)
    return np.array(image), contours

def sample_points_from_contour(contours, num_points=10, use_largest_contour=False, interval=True):
    """
    Sample points from the contour of a mask.

    :param mask: The binary mask from which to extract contours.
    :param num_points: The number of points to sample from the contour.
    :param use_largest_contour: Whether to sample from only the largest contour.
    :return: A list of sampled points (x, y) from the contour.
    """ 
    if not contours:
        return []  # Return an empty list if no contours found

    if use_largest_contour:
        # Find the largest contour based on area
        contours = [max(contours, key=cv2.contourArea)]

    sampled_points = []
    
    
    for contour in contours:
        
        if interval:
            # Calculate the interval for sampling
            interval = len(contour) // num_points
            # Sample points from the contour
            for i in range(0, len(contour), interval):
                point = contour[i][0]  # Contour points are stored as [[x, y]]
                sampled_points.append((point[0], point[1]))
                
                if len(sampled_points) >= num_points:
                    break  # Stop if we have collected enough points
                
        else:
            sampled_points = farthest_point_sampling(contour, num_points-1)

    return sampled_points

def draw_points(image, points, color=(0, 255, 0), object_index=0):
    # Ensure the image is in the correct format
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)


    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        
    for idx, point in enumerate(points):
        height, width = image.shape[:2]
        cv2.circle(image, point, radius=min(height,width)//100+2, color=[0,0,0], thickness=-1)
        cv2.circle(image, point, radius=min(height,width)//100, color=color, thickness=-1)
        text_x = min(point[0] + 5, width - 1)
        text_y = min(point[1] + 5, height - 1)
        text_position = (text_x, text_y)
        cv2.putText(image, f'P{object_index}_{idx}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,0,0], 1)

    return image
##########Done##########


def farthest_point_sampling(contour, num_points=5):
    sampled_points = []
    sampled_points.append(contour[0][0])  # Start with the first point
    for _ in range(1, num_points):
        max_dist = -1
        next_point = None
        for point in contour:
            point = point[0]
            min_dist = np.min([np.linalg.norm(point - np.array(sp)) for sp in sampled_points])
            if min_dist > max_dist:
                max_dist = min_dist
                next_point = point
        sampled_points.append(next_point)
    return sampled_points


def sample_points_from_mask(mask, image, num_points=10, use_largest_contour=True, draw=True,random_color=False, interval=False, other_images=None):
    # Convert mask to a binary image
    image = image.astype(np.uint8)
    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    
    sampled_points_all = []
    centroids = []
    
    fixed_colors=[[30, 144, 255],[255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 255, 0],[0, 255, 255],[255, 0, 255],[255, 255, 255],[0, 0, 0]]
    
    for i in range(mask.shape[0]):
        if random_color:
            color = np.random.random(3) * 255
        else:
            color = fixed_colors[i]
        sub_mask = mask[i][0].numpy()
        binary_mask = sub_mask.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []  # Return an empty list if no contours found

        if use_largest_contour:
            # Find the largest contour based on area
            contours = [max(contours, key=cv2.contourArea)]


        
        for contour in contours:
            sampled_points = []
            M = cv2.moments(contour)
            
            # Check for moment area to be zero to avoid division by zero
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append(np.array((cX, cY),dtype=np.int32))
                sampled_points.append(np.array((cX, cY),dtype=np.int32))
            else:
                centroids.append(np.array((0, 0),dtype=np.int32))

                
            
            if interval:
                # Calculate the interval for sampling
                interval = len(contour) // num_points
                # Sample points from the contour
                for i in range(0, len(contour), interval):
                    point = contour[i][0]  # Contour points are stored as [[x, y]]
                    sampled_points.append((point[0], point[1]))
                    
                    if len(sampled_points) >= num_points:
                        break  # Stop if we have collected enough points
                    
            else:
                sampled_points += farthest_point_sampling(contour, num_points)
            sampled_points_all.append(sampled_points)
            
            
            
            
            # Draw each point
            if draw:
                image_with_points = draw_points(image, sampled_points, color=color, object_index=i)
            
                frames=[image_with_points]
                if other_images is not None:
                    for img in other_images:
                        frames.append(draw_points(img, sampled_points, color=color, object_index=i))
            else:
                frames=None
                    

    
    return sampled_points_all,frames

   



def segmentation_from_text(image_tensor,image_np, text_prompt, grounding_dino_model=groundingdino_model, sam_model=sam_predictor, box_threshold = 0.2, text_threshold = 0.2, VLM_guided=True, VLM_model=None, tokenizer=None, other_images=None):
    
    
    if VLM_guided:
        
        annotated_frame, boxes, response = VLM_guided_detect(image_tensor,image_np, text_prompt, VLM_model,tokenizer,grounding_dino_model, box_threshold, text_threshold)
        num_objects=count_objects(response)
        
        
    else:
        annotated_frame, boxes = detect(image_tensor,image_np, text_prompt, grounding_dino_model, box_threshold, text_threshold)
        
        response=None
            
    if boxes.shape==(0,4):
        print("No object detected")
        return None,Image.fromarray(image_np), Image.fromarray(image_np), response,0
    
    
    segmented_frame_masks = segment(image=image_np, sam_model=sam_model, boxes=boxes)
    # print(segmented_frame_masks.shape)
    if VLM_guided and segmented_frame_masks.shape[0]<num_objects:
        print("Some objects not detected")
        detection_ratio=segmented_frame_masks.shape[0]/num_objects
    elif VLM_guided and segmented_frame_masks.shape[0]>num_objects:
        print("Extra objects detected")
        detection_ratio=num_objects/segmented_frame_masks.shape[0]
        
    elif VLM_guided and segmented_frame_masks.shape[0]==num_objects:
        print("All objects detected")
        detection_ratio=1
        
    else:
        detection_ratio=1
    
    
    sampled_points_all,frame_with_points = sample_points_from_mask(segmented_frame_masks, image_np, num_points=5, use_largest_contour=True, random_color=True, interval=False, other_images=other_images)
    #   annotated_frame_with_mask, contours = draw_mask_contour(segmented_frame_masks[0][0], annotated_frame,)
    #   points=sample_points_from_contour(contours, num_points=10, use_largest_contour=True, interval=False)
    
    #   annotated_frame_with_points = draw_points(annotated_frame_with_mask, points)
    
    
    
    #   annotated_frame_with_mask = sample_points_from_contour(segmented_frame_masks[0][0], num_points=10, use_largest_contour=True, interval=False)
    return sampled_points_all,Image.fromarray(annotated_frame), frame_with_points, response, detection_ratio



# The check_processed_images function can be updated
def check_processed_images(text_description, image):
    system_prompt="""
    Role: You are now being the supervisor to check if all the objects are detected correctly from a VLM. You will be given the objects that the VLM should detect, and the image with the objects detected by the VLM with a box annotator around each detected object. You will need to check if all the objects are detected correctly.
    
    Input Explanation:
    1. Object list: The list of objects that the VLM should detect. Each object is separated by a space and a period. For example, "yellow cup . spoon ." means the VLM should detect a yellow cup and a spoon.
    
    2. Image: The image with the objects detected by the VLM with a box annotator around each detected object.
    
    
    Output Requirement: 
    You will need to check if all the objects are detected correctly. If all the objects are detected correctly, you will need to provide a "Yes" response. If not, you will need to provide a "No" response. Make sure only to provide a "Yes" or "No" response and nothing else. I am using a hard-coded pattern to match your response so please output exactly as requested.
        
    Example output 1 (all objects detected correctly):
    Detect Result: Yes
    
    
    Example output 2 (not all objects detected correctly):
    Detect Result: No
    
    
    """
    
    user_prompt=f"""Here are the objects to detect:
    
    Objects: {text_description}.
    
    And here's the image with the objects detected by the VLM
    
    """
    
    
    messages=[{
        "role":"system",
        "content":system_prompt,
        
    }]
    
    messages.append({
        "role":"user",
        "content":[user_prompt,{"image":numpy_to_base64(image)}],
    })
    
    response=communicate_gpt(messages)
    
    
    # Define the pattern to match "Detect Result: " followed by "Yes" or "No"
    true_pattern = r"Detect Result: Yes"
    false_pattern = r"Detect Result: No"


    matches_yes=re.findall(true_pattern,response)
    matches_no=re.findall(false_pattern,response)
    
    if len(matches_yes)>0:
        return True
    else:
        return False
    
    

def add_grid_to_processed(dataset_folder, grid_size=5, save_to_another_folder=True, target_folder=None):
    sorted_dir=get_sorted_files(dataset_folder,folders=True)
    
    for dir in sorted_dir:
        cur_dir=os.path.join(dataset_folder,dir)
        if save_to_another_folder and target_folder is None:
            cur_target_dir=os.path.join(dataset_folder,dir,"processed_images_with_grid")
            os.makedirs(cur_target_dir,exist_ok=True)
            
        elif save_to_another_folder and target_folder is not None:
            cur_target_dir=os.path.join(target_folder,dir,"processed_images_with_grid")
            os.makedirs(cur_target_dir,exist_ok=True)
            
            
        processed_image_path=os.path.join(cur_dir,"key_points_GPT_guided.png")
        processed_contact_image_path=os.path.join(cur_dir,"contact_points_GPT_guided.png")
        if not os.path.exists(processed_image_path):
            processed_image_path=os.path.join(cur_dir,"key_points_VLM_guided.png")
            processed_contact_image_path=os.path.join(cur_dir,"contact_points_VLM_guided.png")
        
        keyframes_path=get_sorted_files(os.path.join(cur_dir,"key_frames"),folders=False)
        if len(keyframes_path)>2:
            gripper_action=True
            after_contact_image_path=os.path.join(cur_dir,"key_frames",keyframes_path[-2])
        last_frame_path=os.path.join(cur_dir,"key_frames",keyframes_path[-1])
            
        images_to_be_processed=[processed_image_path,processed_contact_image_path,last_frame_path,] 
        if gripper_action:
            images_to_be_processed.insert(2,after_contact_image_path)
        for i,image in enumerate(images_to_be_processed):
            processed_image=Image.open(image)
            processed_image_with_grid=add_grid_to_image(np.array(processed_image), grid_size=grid_size,add_caption=True)
            processed_image_with_grid=Image.fromarray(processed_image_with_grid)
            processed_image_with_grid.save(os.path.join(cur_target_dir,f"processed_frame_{i}.png"))
            
            
        # processed_image=Image.open(processed_image_path)
        # processed_image_with_grid=add_grid_to_image(np.array(processed_image), grid_size=grid_size,add_caption=True)
        # processed_image_with_grid=Image.fromarray(processed_image_with_grid)
        # processed_image_with_grid.save(os.path.join(cur_dir,"key_points_with_grid.png"))
    

def _process_video(video_path,info_path,target_folder,video_index,view_index):
    
    
    # extract frames from video
    episode_dir=os.path.join(target_folder,f"demo_{video_index}")
    os.makedirs(episode_dir,exist_ok=True)
    frames_dir=os.path.join(episode_dir,"frames")
    os.makedirs(frames_dir,exist_ok=True)
    
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0
    
    while success:
        # Save frame as JPEG file
        frame_filename = os.path.join(frames_dir, f"frame_{count}.png")
        cv2.imwrite(frame_filename, image)
        success, image = video_capture.read()
        count += 1

    video_capture.release()
    print(f"Extracted {count} frames from {video_path} to {frames_dir}")
    
    with open(info_path, 'r') as file:
        info_dict = json.load(file)
        task_description=info_dict["instruction"]
        picker_traj=info_dict[f"post_traj_{view_index}"]
        traj_dict={"traj":picker_traj}
    
    with open(os.path.join(episode_dir,f"droid.txt"),"w") as f:
        f.write(task_description)
        
    with open(os.path.join(episode_dir,f"picker_traj.json"),"w") as f:
        json.dump(traj_dict,f,indent=4)
        
        
    video_index+=1
    return video_index
    
    
    
def preprocess_droid(dataset_path="../datasets/droid_junjie"):
    # This function only transformed the droid dataset we have into the format like bridge and jaco_play
    
    passed_check_demo_dir="../datasets/processed_droid"
    os.makedirs(passed_check_demo_dir,exist_ok=True)
    
    valid_video_index=0
    episodes=[f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f!="passed_demos"]
    
    for episode in episodes:
        cur_episode_dir=os.path.join(dataset_path,episode)
        # deal with view 1
        if os.path.exists(os.path.join(cur_episode_dir,"cam1_marked.mp4")):
            valid_video_index=_process_video(video_path=os.path.join(cur_episode_dir,"cam1.mp4"),info_path=os.path.join(cur_episode_dir,"info.json"),target_folder=passed_check_demo_dir,video_index=valid_video_index,view_index=1)
            
        # deal with view 2
        if os.path.exists(os.path.join(cur_episode_dir,"cam2_marked.mp4")):
            valid_video_index=_process_video(video_path=os.path.join(cur_episode_dir,"cam2.mp4"),info_path=os.path.join(cur_episode_dir,"info.json"),target_folder=passed_check_demo_dir,video_index=valid_video_index,view_index=2)


def _get_correct_answer(starting_point,ending_point,use_sampled_points=False,sampled_points=None,img_size=(1280,720),grid_size=5):
    # The correct answer is the distance between the starting point and the ending point
    print(f"Starting point: {type(starting_point)}")
    print(f"Ending point: {type(ending_point)}")
    print(f"Sampled points: {type(sampled_points)}")
    if use_sampled_points:
        sampled_points_cp = np.array(sampled_points)
        starting_point_cp = np.array(starting_point)
        
        distances = np.linalg.norm(sampled_points_cp - starting_point_cp, axis=1)
        closest_point_idx = np.argmin(distances)
        if distances[closest_point_idx] <= 0.1*min(img_size[0],img_size[1]): # make the closet point in the sampled points from the ground truth object is cloth enough to the starting point.
            picking_point = [int(coord) for coord in sampled_points[closest_point_idx]]
        else:
            picking_point = starting_point
            use_sampled_points=False
    else:
        picking_point = starting_point
        
        
    height, width = img_size

    # Calculate the step size for the grid
    x_step = width // grid_size
    y_step = height // grid_size

    starting_tile = [chr(97+(starting_point[0] // x_step)), starting_point[1] // y_step]
    ending_tile = [chr(97+(ending_point[0] // x_step)), ending_point[1] // y_step]

    # if sampled_points is not None:
    #     sampled_points = sampled_points.tolist()
    if type(picking_point)==np.ndarray:
        picking_point=picking_point.tolist()
    correct_answer={
        "picking_point":picking_point,
        "starting_tile":starting_tile,
        "ending_tile":ending_tile,
    }
        
    return correct_answer,use_sampled_points


def _process_image(image,point_dict):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)


    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        
    for idx, point_idx in enumerate(point_dict.keys()):
        color = np.random.random(3) * 255
        height, width = image.shape[:2]
        point=point_dict[point_idx]
        cv2.circle(image, point, radius=min(height,width)//100+2, color=[0,0,0], thickness=-1)
        cv2.circle(image, point, radius=min(height,width)//100, color=color, thickness=-1)
        text_x = min(point[0] + min(height,width)//100, width - 1)
        text_y = min(point[1] + min(height,width)//100, height - 1)
        text_position = (text_x, text_y)
        cv2.putText(image, point_idx, text_position, cv2.FONT_HERSHEY_SIMPLEX, min(height,width)/1000, [0,0,0], 1)
        
        
    image=add_grid_to_image(image, grid_size=5,add_caption=True)
    
    return image


def _get_wrong_answers(episode_dir,number_of_wrong_choice=3,batch_index=0):
    
    
    
    wrong_answer_dir=os.path.join(episode_dir,f"wrong_answers_v_{batch_index}")
    os.makedirs(wrong_answer_dir,exist_ok=True)
    
    with open(os.path.join(episode_dir,"info.json"),"r") as f:
        info=json.load(f)
       
    with open(os.path.join(episode_dir,"correct_answer.json"),"r") as f:
        correct_answer=json.load(f)
    
    # print(info)
    # print(correct_answer)    
    available_picking_points=info["points_idx"]
    confusing_points_index=info["confusing_points_idx"]
    points_to_draw=[correct_answer["picking_point_index"]]
     
    
    wrong_answers={}
    for i in range(number_of_wrong_choice):
        wrong_picking_point_index = random.choice(confusing_points_index)
        wrong_answers_i={"picking_point_index":wrong_picking_point_index}
        points_to_draw.append(wrong_picking_point_index)
        wrong_answers_i["picking_point"]=available_picking_points[wrong_picking_point_index]
        
        starting_tile_x=random.randint(0,4)
        starting_tile_y=random.randint(0,4)
        wrong_answers_i["starting_tile"]=[chr(97+starting_tile_x),starting_tile_y]
        
        ending_tile_x=random.randint(0,4)
        ending_tile_y=random.randint(0,4)
        wrong_answers_i["ending_tile"]=[chr(97+ending_tile_x),ending_tile_y]
        
        wrong_answers[f"answer_{i}"]=wrong_answers_i
        
        
    with open (os.path.join(wrong_answer_dir,f"wrong_answers.json"),"w") as f:
        json.dump(wrong_answers,f,indent=4)        

        
    image_idx=info["detected_frame"]
    if image_idx<=0:
        image_idx=0
    
    image_path=os.path.join(episode_dir,"frames",f"frame_{image_idx}.png")
    image_np,_ = load_image(image_path)
    image_to_process=image_np.copy()
    
    points_dict={}
    for idx in points_to_draw:
        points_dict[idx]=available_picking_points[idx]
        
    processed_image=_process_image(image_to_process,points_dict)
    
    processed_image=Image.fromarray(processed_image)
    
    processed_image.save(os.path.join(wrong_answer_dir,f"processed_frame.png"))
        
        
    
    

def update_question_building_pipeline(dataset_path="../datasets/processed_droid",dataset_name="droid", groundingdino_model=groundingdino_model, sam_model=sam_predictor):
    
    # get all the episode directories in the dataset_path
    sorted_dir=get_sorted_files(dataset_path,folders=True)
    detect_successful=0
    detect_ratio=0
    cases_count=len(sorted_dir)
    
    # make a new directory to store the passed demos
    passed_check_demo_dir=os.path.join(dataset_path,"passed_demos")
    os.makedirs(passed_check_demo_dir,exist_ok=True)
    
    for dir in sorted_dir:# iterate through each episode
        print(f"Processing {dir}")
        cur_dir=os.path.join(dataset_path,dir)
        info={} # store the information of each episode after processing
        # get task description
        with open (os.path.join(cur_dir,f"{dataset_name}.txt"),"r") as f:
            text_prompt=f.read()
        
        
        frame_dir=os.path.join(cur_dir,"frames")
        sorted_images=get_sorted_files(frame_dir,folders=False)
        
        init_image_path=os.path.join(cur_dir,"frames","frame_0.png")
        image_np,image_tensor = load_image(init_image_path)
        # get the object the robot should grasp from task description and the first frame of the video
        ground_truth_object=get_objects_from_text(image_tensor, image_np, text_prompt)
        info["ground_truth_object"]=ground_truth_object
        
        # start detecting the object robot should manipulate
        info["detected_frame"]=0
        detected=True
        annotated_frame,boxes=detect(image_tensor,image_np, ground_truth_object, groundingdino_model, box_threshold=0.3, text_threshold=0.25)
        
        
        if boxes.shape==(0,4): # no object detected
            print("No object detected in this frame")
            detected=False
            for idx,image in enumerate(sorted_images):# iterate through each frame until the object is detected
                image_np,image_tensor = load_image(os.path.join(frame_dir,image))
                annotated_frame,boxes=detect(image_tensor,image_np, ground_truth_object, groundingdino_model, box_threshold=0.3, text_threshold=0.25)
                if boxes.shape!=(0,4):
                    detected=True
                    info["detected_frame"]=idx
                    break
                         
        if detected: # if the object is detected, let GPT decide whether the object is detected correctly from the annotated frame
            check_result=check_processed_images(ground_truth_object, np.array(annotated_frame))
        
        
        Image.fromarray(annotated_frame).save(os.path.join(cur_dir,"detection_result.png")) 
            
        
        segment_needed=detected and check_result # If we can't detect the correct object, we can't segment it correctly and we can't use the object countor.
        threshold=0.1*min(image_np.shape[0],image_np.shape[1]) 
        sampled_points_all=None
        
        if segment_needed: # if we need to perform segmentation on the ground truth object
            
            segmented_frame_masks = segment(image=image_np, sam_model=sam_model, boxes=boxes)
             
            correct_seg=segmented_frame_masks.shape[0]==1 # whether we can segment the object correctly
            
            if correct_seg:
                
                sampled_points_all,_=sample_points_from_mask(segmented_frame_masks, image_np, num_points=5, use_largest_contour=True, draw=False,random_color=True, interval=False)
                
                # Below is we will iterate through each point sampled from the object mask. If the distance between any two points is larger than the threshold, then we can use the points on the object as candidate wrong picking points.
                distance_ok=False
                for idx,point in enumerate(sampled_points_all[0]):
                    for j in range(idx+1,len(sampled_points_all[0])):
                        if np.linalg.norm(np.array(point)-np.array(sampled_points_all[0][j]))>threshold:
                            print("We can use these points")
                            distance_ok=True
                            break
                
                use_sampled_points=True       
            else:
                use_sampled_points=False  
        else:# the object can't be detected correctly throughout the video
            use_sampled_points=False
            info["detected_frame"]=-1
                
        # load the picker's trajectory 
        with open (os.path.join(cur_dir,f"picker_traj.json"),"r") as f:
            picker_traj_dict=json.load(f)
            picker_traj=picker_traj_dict["traj"]
            starting_point=[int(picker_traj[0][0]*4),int(picker_traj[0][1]*4)]
            ending_point=[int(picker_traj[0][0]*4),int(picker_traj[0][1]*4)]
        
        sampled_points_on_object=None if sampled_points_all is None else sampled_points_all[0]
        # get the correct picking point and the correct starting and ending tile
        correct_answer,use_sampled_points=_get_correct_answer(starting_point,ending_point,use_sampled_points,sampled_points_on_object,img_size=(image_np.shape[0],image_np.shape[1]),grid_size=5)     
        

        
        
        
        # get the confusing objects so that we can get confusing points by sampling points from the confusing objects' masks
        list_object_system_prompt=f"""
        
        Role: You need to list 2 objects in the image and make sure the objects you listed are clear, unocculuded, and easy to spot. It will also be nice if the objects are not too close to each other.
        """
        
        list_object_user_prompt=f"""
        
        I am giving you an image. Please list 2 objects except {ground_truth_object}. And output the objects with a space and a period between each object.
        
        For example, if you see a yellow cup, a spoon, {ground_truth_object} you should output:
        
        yellow cup . spoon .
        
        """
        
        confusing_objects=get_objects_from_text(image_tensor, image_np, text_prompt, system_prompt=list_object_system_prompt, user_prompt=list_object_user_prompt)
        info["confusing_objects"]=confusing_objects
        
        
        # detect the confusing objects and sample points from the confusing objects' masks
        annotated_confusing_objects, confusing_boxes = detect(image_tensor,image_np, confusing_objects, groundingdino_model, box_threshold=0.3, text_threshold=0.25) 
            
        detected_confusing_objects_count=confusing_boxes.shape[0]
        
        if confusing_boxes.shape[0]>0:
            segmented_frame_masks = segment(image=image_np, sam_model=sam_model, boxes=confusing_boxes)
            sampled_points_all_confusing,_ = sample_points_from_mask(segmented_frame_masks, image_np, num_points=5, use_largest_contour=True, draw=False,random_color=True, interval=False)
            
            print(f"sampled_points_all_confusing (how many objects detected): {len(sampled_points_all_confusing)}")
            if len(sampled_points_all_confusing)>=2:
                sampled_points_all_confusing=sampled_points_all_confusing[:2]
            
        
            random_sampling_points_for_confusion=2-len(sampled_points_all_confusing)
        else:
            random_sampling_points_for_confusion=2
            sampled_points_all_confusing=[]
        
        available_picking_points={}# all the points
        available_picking_points_coordinate_list=[]
        confusing_points=[]
        # for random_index in range(random_sampling_points):
        #     random_point=[np.random.randint(0,image_np.shape[1]),np.random.randint(0,image_np.shape[0])]
        #     available_picking_points_coordinate_list.append(random_point)
        
        
        if use_sampled_points: # If we decide to use the points on the object mask as candidate wrong picking points
            print(sampled_points_on_object)
            for point in sampled_points_on_object:# The points on the object mask
                
                available_picking_points_coordinate_list.append(point)
                print(f"point {point}")
                if (point[0]!=correct_answer["picking_point"][0] or point[1]!=correct_answer["picking_point"][1] ) and distance_ok: # Only include them as candidate wrong picking points if they are not the correct picking point and there are two points on the object mask that are far enough from each other  
                    confusing_points.append(point)
                
        else:
            available_picking_points_coordinate_list.append(correct_answer["picking_point"])# There are no points sampled from the object mask
            for i in range(4):# Get 4 random points within a distance of 0.1*min(image_np.shape[0],image_np.shape[1]) from the correct picking point as the confusing points
                random_x_low=max(correct_answer["picking_point"][0]-threshold,0)
                random_x_high=min(correct_answer["picking_point"][0]+threshold,image_np.shape[1])
                
                mid_x_1=max(correct_answer["picking_point"][0]-0.5*threshold,0)
                mid_x_2=min(correct_answer["picking_point"][0]+0.5*threshold,image_np.shape[1])
                
                combined_x_range=list(range(int(random_x_low),int(mid_x_1)))+list(range(int(mid_x_2),int(random_x_high)))
                
                random_y_low=max(correct_answer["picking_point"][1]-threshold,0)
                random_y_high=min(correct_answer["picking_point"][1]+threshold,image_np.shape[0])
                
                mid_y_1=max(correct_answer["picking_point"][1]-0.5*threshold,0)
                mid_y_2=min(correct_answer["picking_point"][1]+0.5*threshold,image_np.shape[0])
                
                combined_y_range=list(range(int(random_y_low),int(mid_y_1)))+list(range(int(mid_y_2),int(random_y_high)))
                
                
                random_point_x=random.choice(combined_x_range)
                random_point_y=random.choice(combined_y_range)
                
                random_point=[random_point_x,random_point_y]
                
                available_picking_points_coordinate_list.append(random_point)
                confusing_points.append(random_point)
                
        for points in sampled_points_all_confusing:
            for point in points:
                available_picking_points_coordinate_list.append(point)
                confusing_points.append(point)

        for i in range(random_sampling_points_for_confusion):
            for j in range(5):
                
                
                mid_x_1=max(correct_answer["picking_point"][0]-0.5*threshold,0)
                mid_x_2=min(correct_answer["picking_point"][0]+0.5*threshold,image_np.shape[1])
                
                combined_x_range=list(range(0,int(mid_x_1)))+list(range(int(mid_x_2),image_np.shape[1]))
                
                
                mid_y_1=max(correct_answer["picking_point"][1]-0.5*threshold,0)
                mid_y_2=min(correct_answer["picking_point"][1]+0.5*threshold,image_np.shape[0])
                
                combined_y_range=list(range(0,int(mid_y_1)))+list(range(int(mid_y_2),image_np.shape[0]))
                
                
                random_point_x=random.choice(combined_x_range)
                random_point_y=random.choice(combined_y_range)
                
                random_point=[random_point_x,random_point_y]
                
                available_picking_points_coordinate_list.append(random_point)
                confusing_points.append(random_point)
                
                
                
                
                

                 
            
        confusing_points_index=[]
        random.shuffle(available_picking_points_coordinate_list)
        for idx,point in enumerate(available_picking_points_coordinate_list):
            available_picking_points[f"p_{idx}"]=[int(coord) for coord in point]
            if point[0]==correct_answer["picking_point"][0] and point[1]==correct_answer["picking_point"][1]:
                correct_picking_point_index=f"p_{idx}"
            
            for confusing_point in confusing_points:
                if point[0]==confusing_point[0] and point[1]==confusing_point[1]:
                    confusing_points_index.append(f"p_{idx}")
  
  
  
  
        info["points_idx"]=available_picking_points
        info["confusing_points_idx"]=confusing_points_index
        
        
        correct_answer["picking_point_index"]=correct_picking_point_index
        with open (os.path.join(cur_dir,f"correct_answer.json"),"w") as f:
            json.dump(correct_answer,f,indent=4)
        
        with open(os.path.join(cur_dir,f"info.json"),"w") as f:
            json.dump(info,f,indent=4)
        
        
        
        
        wrong_answers={}
        for i in range(3):
            wrong_picking_point_index = random.choice(confusing_points_index)
            wrong_answers_i={"picking_point_index":wrong_picking_point_index}
            wrong_answers_i["picking_point"]=available_picking_points[wrong_picking_point_index]
            
            starting_tile_x=random.randint(0,4)
            starting_tile_y=random.randint(0,4)
            wrong_answers_i["starting_tile"]=[chr(97+starting_tile_x),starting_tile_y]
            
            ending_tile_x=random.randint(0,4)
            ending_tile_y=random.randint(0,4)
            wrong_answers_i["ending_tile"]=[chr(97+ending_tile_x),ending_tile_y]
            
            wrong_answers[f"answer_{i}"]=wrong_answers_i
            
           
        with open (os.path.join(cur_dir,f"wrong_answers.json"),"w") as f:
            json.dump(wrong_answers,f,indent=4)        

            
        
        
        
        image_to_process=image_np.copy()
        processed_image=_process_image(image_to_process,available_picking_points)
        
        processed_image=Image.fromarray(processed_image)
        
        processed_image.save(os.path.join(cur_dir,f"all_points_drawn.png"))
    # print(f"Done. {detect_successful} out of {cases_count} cases detected successfully. Average detection ratio: {detect_ratio/cases_count}")
    # print("===========================================")
    # print(f"Begin adding the grid to the processed images")
    
    # add_grid_to_processed(passed_check_demo_dir, grid_size=5)



def generate_wrong_answers(dataset_path="../datasets/processed_droid",dataset_name="droid",batch_index=0):
    # get all the episode directories in the dataset_path
    sorted_dir=get_sorted_files(dataset_path,folders=True)
    
    # make a new directory to store the passed demos
    passed_check_demo_dir=os.path.join(dataset_path,"passed_demos")
    os.makedirs(passed_check_demo_dir,exist_ok=True)
    
    for dir in sorted_dir:# iterate through each episode
        print(f"Processing {dir}")
        cur_dir=os.path.join(dataset_path,dir)
        
        _get_wrong_answers(cur_dir,batch_index=batch_index)
        
        
        
if __name__ == "__main__":
    

    
    def test_on_dataset(dataset_path="../datasets/jaco_play",save_processed_image=True,dataset_name="jaco_play", VLM_guided=False, VLM_model=None,tokenizer=None):
        sorted_dir=get_sorted_files(dataset_path,folders=True)
        detect_successful=0
        detect_ratio=0
        cases_count=len(sorted_dir)
        
        passed_check_demo_dir=os.path.join(dataset_path,"passed_demos")
        os.makedirs(passed_check_demo_dir,exist_ok=True)
        
        for dir in sorted_dir:
            
            cur_dir=os.path.join(dataset_path,dir)
            key_frame_dir=os.path.join(cur_dir,"key_frames")
            
            sorted_images=get_sorted_files(key_frame_dir,folders=False)
            
            gripper_action=len(sorted_images)>2
            
            other_images=[]
            for idx,path in enumerate(sorted_images):
                if idx>=1:
                    image,_=load_image(os.path.join(key_frame_dir,path))
                    other_images.append(np.array(image))
                
                
            init_image_path=os.path.join(cur_dir,"key_frames","frame_0.png")
            
            
            with open (os.path.join(cur_dir,f"{dataset_name}.txt"),"r") as f:
                text_prompt=f.read()
            image_np,image_tensor = load_image(init_image_path)
            
            sampled_points,annotated_frame,frames_with_points,response,detection_ratio = segmentation_from_text(image_tensor,image_np, text_prompt, box_threshold=0.3, text_threshold=0.25, VLM_guided=VLM_guided,VLM_model=VLM_model,tokenizer=tokenizer,other_images=other_images)
        
            if sampled_points is None:
                continue
            
            annotated_frame_points=Image.fromarray(frames_with_points[0])
            annotated_frame_contact_points=Image.fromarray(frames_with_points[1])
            
            if VLM_guided and VLM_model is not None:
                annotated_frame.save(os.path.join(cur_dir,"processed_image_VLM_guided.png"))
                
                annotated_frame_points.save(os.path.join(cur_dir,"key_points_VLM_guided.png"))
                
                annotated_frame_contact_points.save(os.path.join(cur_dir,"contact_points_VLM_guided.png"))
                with open (os.path.join(cur_dir,f"VLM_response.txt"),"w") as f:
                    f.write(response)
            elif VLM_guided and VLM_model is None:
                annotated_frame.save(os.path.join(cur_dir,"processed_image_GPT_guided.png"))
                annotated_frame_points.save(os.path.join(cur_dir,"key_points_GPT_guided.png"))
                annotated_frame_contact_points.save(os.path.join(cur_dir,"contact_points_GPT_guided.png"))
                with open (os.path.join(cur_dir,f"GPT_response.txt"),"w") as f:
                    f.write(response)
            else:
                annotated_frame.save(os.path.join(cur_dir,"processed_image_test.png"))
                annotated_frame_points.save(os.path.join(cur_dir,"key_points.png"))
                annotated_frame_contact_points.save(os.path.join(cur_dir,"contact_points.png"))
                
                
            if detection_ratio==1:
                
                check_result=check_processed_images(response, np.array(annotated_frame))
                
                if check_result:
                    detect_successful+=1
                    destination_dir = os.path.join(passed_check_demo_dir, dir)
                    shutil.copytree(cur_dir, destination_dir)
                
            diff_ratio=np.abs(1-detection_ratio)
            detect_ratio+=(1-diff_ratio)
                

        
        print(f"Done. {detect_successful} out of {cases_count} cases detected successfully. Average detection ratio: {detect_ratio/cases_count}")
        print("===========================================")
        print(f"Begin adding the grid to the processed images")
        
        add_grid_to_processed(passed_check_demo_dir, grid_size=5)
        
        
        # sorted_dir_passed=get_sorted_files(passed_check_demo_dir,folders=True)
        
        # for dir in sorted_dir_passed:
            
        
    
          
    # update_question_building_pipeline(dataset_path="../datasets/processed_droid")
    generate_wrong_answers(dataset_path="../datasets/processed_droid",batch_index=0)
    
    
    
    
    # preprocess_droid() # to be comment after preprocess
    
    
    
    
    
       
    # parser=argparse.ArgumentParser() 
    # # parser.add_argument("--dataset_path", type=str, default="../datasets/jaco_play", help="Path to the dataset")
    # parser.add_argument("--save_processed_image", type=bool, default=True, help="Save the processed image")
    # parser.add_argument("--dataset_name", type=str, default="jaco_play", help="Name of the dataset")
    # parser.add_argument("--VLM_guided", type=int, default=0, help="Use VLM guided detection")
    # parser.add_argument("--VLM_model", type=str, default="GPT")
    # args=parser.parse_args()
    
    # dataset_path=f"../rtx_datasets/{args.dataset_name}"
    # VLM_guided=bool(args.VLM_guided)
    
    # if args.VLM_model=="GPT":
    #     VLM_model=None
    #     tokenizer=None
    # else:
    #     # Load VLM
    #     GLM_path="THUDM/glm-4v-9b"
    #     tokenizer=AutoTokenizer.from_pretrained(GLM_path, trust_remote_code=True)
    #     quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    #     try:
    #         glm_4v=AutoModelForCausalLM.from_pretrained(GLM_path, quantization_config=quantization_config,device_map=DEVICE_0, trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
    #         model=glm_4v
    #     except torch.cuda.OutOfMemoryError:
    #         model=None

    #     VLM_model=model
        
        
    # test_on_dataset(dataset_path=dataset_path,save_processed_image=True,dataset_name=args.dataset_name, VLM_guided=VLM_guided,VLM_model=VLM_model,tokenizer=tokenizer)