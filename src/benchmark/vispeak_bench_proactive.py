import tqdm
import os
import time
import json
from utils.data_execution import get_model_response
from utils.video_execution import split_video

from benchmark.benchmark import Benchmark



EVAL_CONFIG = {
    "aw":{"anno_file":"annotations/Anomaly_Warning.json", "time_acc": True, "text_score":True}, 
    "gu":{"anno_file":"annotations/Gesture_Understanding.json", "time_acc": True, "text_score":True}, 
    "hr":{"anno_file":"annotations/Humor_Reaction.json", "time_acc": True, "text_score":True}, 
    "vi":{"anno_file":"annotations/Visual_Interruption.json", "time_acc": True, "text_score":False}, 
    "vr":{"anno_file":"annotations/Visual_Reference.json", "time_acc": False, "text_score":False},  
    "vt":{"anno_file":"annotations/Visual_Termination.json", "time_acc": True, "text_score":True},  
    "vw":{"anno_file":"annotations/Visual_Wake-Up.json", "time_acc": True, "text_score":True},  
}




class VispeakBenchProactive(Benchmark):
    def __init__(self, data_file, video_root):
        self.data_file = data_file
        self.video_root = video_root


    def get_task_promptv2(self, stage):
        if any([task in self.data_file for task in ["Gesture"]]):
            if stage==1:
                return "You're watching a video. At this moment in the video, is there any gesture being made in the video? You can only answer yes or no"
            else:
                return "What gesture did the person in the video make, and what does it signify when considering the context of the preceding conversation?"
        elif any([task in self.data_file for task in ["Termination", "Wake-Up", "Interruption"]]):
            if stage==1:
                if "Wake-Up" in self.data_file:
                    return "You're watching a video. At this moment in the video, is there any gesture/action being made in the video? You can only answer yes or no"
                elif "Termination" in self.data_file:
                    return "You're watching a video. At this moment in the video, is there any gesture/action being made in the video? You can only answer yes or no"
                elif "Interruption" in self.data_file:
                    return "You're watching a video. At this moment in the video, is there any gesture/action being made in the video? You can only answer yes or no"

            else:
                if "Wake-Up" in self.data_file:
                    return "When you see greeting gesture, what should you respond to me? Directly output your response."
                elif "Termination" in self.data_file:
                    return "When you see the goodbye gesture, what should you respond to me? Directly output your response."
                elif "Interruption" in self.data_file:
                    return "When you see the body language or gesture that indicates interruption, you should say stop. What should you respond to me now? Directly output your response."
                # return "When you see the gesture/action of the person in video, what does it mean, and how should you respond? Please directly output your response."
                #return "what is my gesture or action as you see? Is there anything more you would like to say?"
        elif "Anomaly" in self.data_file:
            if stage==1:
                return "You're watching a video. At this moment in the video, is there anything unusual happening in the video? You can only answer yes or no"
            else:
                return "What unusual events occur in this video, and what is your suggestion based on these observations?"
        elif "Humor" in self.data_file:
            if stage==1:
                return "You're watching a video. At this moment in the video, is there anything funny happening in the video? You can only answer yes or no"
            else:
                return "What interesting events occurred in the video, and why?"
       
        

    def eval(self, data, model, output_path):
        results = []
        for info in tqdm.tqdm(data):
            
          
            video_path = info["video"]

            last_gpt_answer = info["conversations"][-1]
            last_gpt_answer_text = info["conversations"][-1]['value']

            if "timespan" in last_gpt_answer:
                ref_time = last_gpt_answer["timespan"]
                max_time = ref_time[1] + 2
            elif "time" in last_gpt_answer:
                ref_time = last_gpt_answer["time"]
                max_time = ref_time + 2 # Maximum polling time: ground truth + 2 seconds
            else:
                ref_time = None
                max_time = None

            start_time = info.get("video_start_time", 0)
            
            dialog_history = []
            answered = False
            # Prepare input for the model
            
            current_time = start_time + 1 # at least 1 frame for inferring
            # current_time = max(ref_time[0]-2, start_time+1)  # debug 

            query = [("human", self.get_task_promptv2(stage=1), current_time)]

            while current_time <= max_time:

                interval = 1

                file = os.path.join(self.video_root, video_path)

                # model inference
                time_s = time.time()
                # print(start_time, current_time)
                response = model.Run(file, query, start_time, current_time)
               
                time_e = time.time()
                timecost = time_e - time_s

                # Record the interaction
                dialog_history.append({
                    'role': 'user', 'content': query, 'time': current_time, 'cost': timecost
                })
                dialog_history.append({
                    'role': 'assistant', 'content': response, 'time': current_time, 'cost': timecost
                })

                response2 = None
                if 'yes' in response.strip().lower():
                    
                    time_s = time.time()
                    if any([task in self.data_file for task in ["Anomaly", "Humor", "Wake-Up"]]):
                        query2 = [("human", self.get_task_promptv2(stage=2), current_time)]
                    else:
                        query2 = [("human", info["conversations"][0]['value'], info["conversations"][0]['time']), 
                                    ("gpt", info["conversations"][1]['value'], info["conversations"][1]['time']), 
                                    ("human", self.get_task_promptv2(stage=2), current_time)]

                    response2 = model.Run(file, query2, start_time, current_time) 
                    time_e = time.time()
                    timecost = time_e - time_s

                    # Record the interaction
                    dialog_history.append({
                        'role': 'user', 'content': query2, 'time': current_time, 'cost': timecost
                    })
                    dialog_history.append({
                        'role': 'assistant', 'content': response2, 'time': current_time, 'cost': timecost
                    })

                    answered = current_time
                    break

                current_time += interval

            results.append({video_path:[response2, current_time, dialog_history]})
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
    
