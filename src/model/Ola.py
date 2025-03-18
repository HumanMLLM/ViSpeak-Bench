
import re
import os
import torch
import copy
from model.modelclass import Model

import argparse
import json
import os
import random
import torch
from tqdm import tqdm
import re
import copy

import moviepy.editor as mp
from typing import Dict, Optional, Sequence, List
import librosa
import whisper

os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

from ola.conversation import conv_templates, SeparatorStyle
from ola.model.builder import load_pretrained_model
from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token, tokenizer_speech_token
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX



from utils.video_execution  import _get_rawvideo_dec

# DEFAULT_IMAGE_TOKEN = '<image>'
# IMAGE_TOKEN_INDEX = -200




class OLA(Model):
    def __init__(self, model_path="THUdyh/Ola-7b"):
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None)
        self.model = model.to('cuda').eval()
        self.model = model.bfloat16()

        self.conv_mode = "qwen_1_5"
        self.image_tokenizer = tokenizer_image_token
        
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

       

    def Run(self, file, inp, start_time, end_time):
            

      
            patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time, video_time = _get_rawvideo_dec(file, 
                            self.image_processor, max_frames=64, video_framerate=1,  min_frames=4,
                            start_time=start_time, end_time=end_time, image_aspect_ratio='square')
            patch_images = torch.stack(patch_images).bfloat16().cuda()

            speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
            speech_lengths = [torch.LongTensor([3000]).to('cuda')]
            speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
            speech_chunks = [torch.LongTensor([1]).to('cuda')]

            patch_images  = patch_images.unsqueeze(0)
            video_processed = (patch_images, patch_images)

            video_data = (video_processed, (384, 384), ["video"])


            conv = copy.deepcopy(conv_templates[self.conv_mode])
            conv.messages = []
            content  = ""

            # content += DEFAULT_IMAGE_TOKEN*patch_images.size(0)+ "\n" + inp

            # conv.append_message(conv.roles[0], content)
            # conv.append_message(conv.roles[1], None)


            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN*patch_images.size(0)+ "\n")
            for role, mes, _ in inp:
                if role == "human":
                    conv.append_message(conv.roles[0], mes)
                elif role =="gpt":
                    conv.append_message(conv.roles[1], mes)

            # content += DEFAULT_IMAGE_TOKEN*patch_images.size(0)+ "\n" + inp

            # conv.append_message(conv.roles[0], content)
            conv.append_message(conv.roles[1], None)


            prompt_question = conv.get_prompt()
            # print("**"*10)
            print(prompt_question)
            input_ids = self.image_tokenizer(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).cuda()

    
            pad_token_ids = 151643
            attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            gen_kwargs = {}

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1


            cont = self.model.generate(
                    inputs=input_ids,
                    images=video_data[0][0],
                    images_highres=video_data[0][1],
                    modalities=video_data[2],
                    speech=speechs,
                    speech_lengths=speech_lengths,
                    speech_chunks=speech_chunks,
                    speech_wav=speech_wavs,
                    attention_mask=attention_masks,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
          
            text_response = text_outputs
            print(text_response)
            return text_response


    