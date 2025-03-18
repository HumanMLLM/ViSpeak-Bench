from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import os
from utils.video_execution import split_video

model, processor = None, None

from model.modelclass import Model


from model.modelclass import Model
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from decord import VideoReader, cpu    # pip install decord
import numpy as np

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def encode_video(video_path, 
            cache_dir,
            start_time,
            end_time,
            max_frames=16,
            min_frames=4,
            video_framerate=3
        ):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_cache_path = os.path.join(cache_dir, video_name)

    if not os.path.exists(video_cache_path):
        os.makedirs(video_cache_path)

    vreader = VideoReader(video_path, ctx=cpu(0))
    fps = vreader.get_avg_fps()
    video_time = len(vreader)/fps
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    img_paths = []
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            # sample_pos = np.linspace(f_start, f_end, num=min_frames, dtype=int)
            # sample_pos = sample_pos.tolist()
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
    
    for i, img in enumerate(patch_images):
        frame_number = sample_pos[i]
        img_path = os.path.join(video_cache_path, f"{video_name}_fps{fps}_frame{frame_number}.png")
        if not os.path.exists(img_path):
            img.save(img_path)

        img_paths.append(img_path)

    

    return img_paths

class Qwen2P5VL(Model):
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)


    def Run(self, file, inp, start_time, end_time):
               
        image_paths = encode_video(file, './vispeak_bench_cache',start_time, end_time, video_framerate=1)
        content = []

        for img_pth in image_paths:
            content.append({"type": "image","image":img_pth, "max_pixels": 360 * 420})

       
        messages = [
                {
                    "role": "user",
                    "content": content
                    
                }
            ]
        for role, mes, _ in inp:
            if role == "human":
                messages.append({"role": "user", "content": mes})
            else:
                            
                messages.append({"role": "assistant", "content": mes})

        
       
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        print(response)
        return response
    
    def name(self):
        return "Qwen2.5-VL"
