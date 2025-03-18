from tqdm import tqdm
import os
import json
from utils.data_execution import get_model_response
from utils.video_execution import split_video

from benchmark.benchmark import Benchmark



class VispeakBench(Benchmark):
    def __init__(self, data, video_root):
        
        self.video_root = video_root

    def eval(self, data, model, output_path):
        results = []
        for info in tqdm(data):
            question = info["questions"]
            video_path = info["video"]
            start_time = info["video_start_time"]
            end_time = info["video_end_time"]
         

            file = os.path.join(self.video_root, video_path)

            response = model.Run(file, [("human",question,10000)], start_time, end_time)
            results.append({video_path:[response, 0]})
           
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
