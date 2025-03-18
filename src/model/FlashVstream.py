import requests
from decord import VideoReader, cpu
import torch
from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flash_vstream.conversation import conv_templates, SeparatorStyle
from flash_vstream.model.builder import load_pretrained_model
from flash_vstream.utils import disable_torch_init
from flash_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from torch.multiprocessing import Process, Queue, Manager
from transformers import TextStreamer
from utils.video_execution  import _get_rawvideo_dec

from model.modelclass import Model
class FlashVstream(Model):
    def __init__(self, model_path="IVGSZ/Flash-VStream-7b"):

        model_name = get_model_name_from_path(model_path)
        model_base = None
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device="cuda", device_map="auto")
        print("Model initialized.")

        self.tokenizer  = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len= context_len
        
    def Run(self, file, inp, start_time, end_time):
              
        patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time, video_time = _get_rawvideo_dec(file, 
                        self.image_processor, max_frames=64, video_framerate=3,  min_frames=4,
                        start_time=start_time, end_time=end_time, image_aspect_ratio='square')
        video = torch.stack(patch_images).half().cuda()
        # video = load_video(file)
        # video = self.image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
        video = [video]


        conv = conv_templates["vicuna_v1"].copy()

        if self.model.config.mm_use_im_start_end:
            img_content = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
        else:
            img_content = DEFAULT_IMAGE_TOKEN + "\n"

        for i,(role, mes, _) in enumerate(inp):
            if role == "human":
                if i == 0:
                    conv.append_message(conv.roles[0], img_content+mes)
                else:
                    conv.append_message(conv.roles[0], mes)
            elif role =="gpt":
                conv.append_message(conv.roles[1], mes)

        conv.append_message(conv.roles[1], None)


        prompt = conv.get_prompt()

        # print("**"*10)
        # print(prompt)
       

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video,
                do_sample=True,
                temperature=0.002,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
        input_token_len = input_ids.shape[1]
            
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print(outputs)
        return outputs

    
    def name(self):
        return "Flash-VStream"


    
