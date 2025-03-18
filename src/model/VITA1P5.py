
import re
import os
import torch
import copy
from model.modelclass import Model

from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
)


from vita.model.builder import load_pretrained_model
from vita.conversation import conv_templates
from vita.util.mm_utils import get_model_name_from_path, tokenizer_image_token


from utils.video_execution  import _get_rawvideo_dec

DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200

class VITA(Model):
    def __init__(self, model_path="VITA-MLLM/VITA-1.5"):
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path, None, model_name, 'qwen2p5_instruct'
        )
        self.conv_mode = 'qwen2p5_instruct'

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        self.image_processor = vision_tower.image_processor

        audio_encoder = model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        self.audio_processor = audio_encoder.audio_processor

        self.image_tokenizer = tokenizer_image_token
        
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

       

    def Run(self, file, inp, start_time, end_time):
            
        print(file)
        pooling_size = getattr(self.model.config, "pooling_size", 1)
        patch_images, slice_len, audio_patch, all_num_audio_seg, sample_time, video_time = _get_rawvideo_dec(file, 
                        self.image_processor, max_frames=16, video_framerate=1, 
                        start_time=start_time, end_time=end_time, image_aspect_ratio=self.model.config.image_aspect_ratio)
        patch_images = torch.stack(patch_images).half().cuda()
        img_token_num = 256

        audios = dict()
        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audios['audios'] = audio.half().cuda()
        audios['lengths'] = audio_length.half().cuda()
        audio_for_llm_lens = 60
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

        all_num_audio_seg = [0 for _ in all_num_audio_seg]
        audio_patch = []
        video_audios = dict()
        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        video_audios['audios'] = audio.half().cuda()
        video_audios['lengths'] = audio_length.half().cuda()
        audio_for_llm_lens = 60
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        video_audios["lengths_for_llm"] = audio_for_llm_lens.cuda()



        conv = copy.deepcopy(conv_templates[self.conv_mode])
        conv.messages = []
        content  = ""

        # print(inp)
        # print(patch_images.size(0))
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN*patch_images.size(0)+ "\n")
        for role, mes, _ in inp:
            if role == "human":
                conv.append_message(conv.roles[0], mes)
            elif role =="gpt":
                conv.append_message(conv.roles[1], mes)

        # content += DEFAULT_IMAGE_TOKEN*patch_images.size(0)+ "\n" + inp

        # conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)

        modality = 'video'
        

        prompt_question = conv.get_prompt(modality)
        # print("**"*10)
        # print(prompt_question)
        input_ids = self.image_tokenizer(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()


        keywords = ['<|im_end|>']
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        sf_masks = torch.tensor([0]*len(patch_images)).cuda()


        cont = self.model.generate(
        input_ids,
        images=patch_images,
        audios=audios,
        sf_masks=sf_masks,
        do_sample=False,
        temperature=0.01,
        max_new_tokens=2048,
        stopping_criteria=[stopping_criteria],
        shared_v_pid_stride=None#2#16#8#4#1#None,
                )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        if '☞' in text_outputs or '☜' in text_outputs or '☟' in text_outputs:
            text_response = text_outputs[1:]
        else:
            text_response = text_outputs
        print(text_response)
        return text_response


