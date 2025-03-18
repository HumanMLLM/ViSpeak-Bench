import json
import csv
from collections import defaultdict
import argparse
import os
import requests
import time
import re
from tqdm import tqdm


url = os.environ["API"] 
token = os.environ["API_KEY"] 


TERMINATE = """
Determine if the GPT text contains intent to conclude the conversation. 
Use a 0-5 scale where 0 indicates no termination signals and 5 shows explicit closure intent.
5 = Contains explicit closure phrases (e.g., "Goodbye", "End of discussion")
4 = Implies conclusion with summary + sign-off (e.g., "In summary...feel free to ask more")
3 = Neutral statement with closing tendency (e.g., "Let me know if needed")
2 = Ambiguous transitional phrasing (e.g., "Anyway...")
1 = Contains weak closure cues (e.g., single emoji/period)
0 = No termination indicators, continues dialog flow

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.
DO NOT INCLUDE ANY MARKDOWN FORMAT.
Only provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}

"""

WAKE_UP = """
Determine if the GPT text expresses greeting intent. 
Use a 0-5 scale where 0 indicates no greeting signals and 5 shows explicit welcoming language.
5 = Explicit welcoming phrases ("Hello!", "How can I assist?")
4 = Contextual opening (Matching time-based greetings like "Good morning")
3 = Service-oriented initiation ("What can I help with?")
2 = Neutral conversational starters ("Shall we begin?")
1 = Implicit opening cues (Single welcoming emoji)
0 = No greeting/dialogue initiation detected

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.
DO NOT INCLUDE ANY MARKDOWN FORMAT.
Only provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}

"""

INTERRUPT="""
Determine if the GPT text contains cessation intent (stopping speaking/concluding). 
Use a 0-5 scale where 0 indicates continued engagement and 5 shows explicit termination.
5 = Explicit acknowledgment ("I'll stop here") + closure
4 = Implicit cessation recognition ("Unless you need more...") 
3 = Passive-agreement cessation ("As requested, ending now")
2 = Ambiguous continuation ("...but") after stop cue
1 = Partial acknowledgment without closure
0 = Complete continuation disregard

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.
DO NOT INCLUDE ANY MARKDOWN FORMAT.
Only provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}

"""

ANOMALY="""
Evaluate video anomaly response from GPT with the following metric, taking into account the total score of 5 points, 
with separate scores for Description Consistency between ground truth and Advice Rationality:

Description Consistency between Ground Truth:
3 = Core elements match without errors
2 = Core elements match with minor errors
1 = Superficial match
0 = Key element errors or contradictory

Advice Rationality:
2 = Actionable & Safe & Logically sound
1 = Generally appropriate
0 = Dangerous/hallucinated

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.
DO NOT INCLUDE ANY MARKDOWN FORMAT.
Only provide the Total score and reason in JSON format. For example, {"description":3, "advice":2, "reason": "this is a reason for the score"}

"""


HUMOR="""
Evaluate alignment between Ground Truth and GPT Text regarding humorous event descriptions. 
5 = Perfect match in humor and delivery
4 = Preserves main humor, but with minor changes to the story or details
3 = Only partial humor retention with some deviations
2 = Only partial humor retention and some important parts are missing
1 = Superficial similarity only
0 = No comedic correlation

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.
DO NOT INCLUDE ANY MARKDOWN FORMAT.
Only provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}

"""

GESTURE="""
Evaluate gesture response from GPT with the following metric, taking into account the total score of 5 points, 
with separate scores for gesture recognition and contextual appropriateness of the response:

Gesture recognition:
3 = Precise gesture identification
2 = Ambiguous gesture reference
1 = No explicit mention of gestures
0 = Hallucinated/non-existent gesture

Contextual appropriateness:
2 = Natural integration with dialogue
1 = Generic but relevant response
0 = Irrelevant/contradictor response

[Dialogue History] provided for context
[Gesture] is the ground truth
[Contextual Reference Text] as a reference, but does not have to match exactly

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.
DO NOT INCLUDE ANY MARKDOWN FORMAT.
Only provide the score and reason in JSON format. For example, {"gesture":3, "context":2, "reason": "this is a reason for the score"}

"""



EVAL_CONFIG = {
    "aw":{"anno_file":"annotations/Anomaly_Warning.json", "time_acc": True, "text_score":True}, 
    "gu":{"anno_file":"annotations/Gesture_Understanding.json", "time_acc": True, "text_score":True}, 
    "hr":{"anno_file":"annotations/Humor_Reaction.json", "time_acc": True, "text_score":True}, 
    "vi":{"anno_file":"annotations/Visual_Interruption.json", "time_acc": True, "text_score":True}, 
    "vr":{"anno_file":"annotations/Visual_Reference.json", "time_acc": False, "text_score":False},  
    "vt":{"anno_file":"annotations/Visual_Termination.json", "time_acc": True, "text_score":True},  
    "vw":{"anno_file":"annotations/Visual_Wake-Up.json", "time_acc": True, "text_score":True},  
}


def gpt_score(prompt):
    success = False
    max_try = 10
    tries = 0
    response_message = 0.0
    while (not success and tries <max_try):
        try:
            messages = [
               
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                     
                    ]
                    
                }
            ]
            data = {
                    "model": "gpt-4o-2024-05-13",
                    "messages":messages,
                    # "n": 1
                }
            
            headers = {
                    "Content-Type": "application/json",
                        "Authorization": 'Bearer ' + token}
            response = requests.post(url, json=data, headers=headers)

            response = response.json()
            response_message = response['choices'][0]['message']['content']

            jsonfy_result = json.loads(response_message)
           
            success = True
        except Exception as e:
            print(f'{e}')
            time.sleep(1)
            tries +=1

    print(jsonfy_result)
    return jsonfy_result

def in_time(response_time, ref_time, margin=3.0):
    if isinstance(ref_time, (tuple, list)):
        start, end = ref_time[0], ref_time[1]
        ex_start = max(0.0, start)
        ex_end = end + margin

        if ex_start <= response_time <= ex_end:
            return True
        else:
            return False

    else:
        if response_time> ref_time or abs(- ref_time) <= margin:
            return True
        else:
            return False
    
    

def extract_characters_regex(s, choices=['A', 'B', 'C', 'D', 'E']):
    if type(s) is dict:
        s = ''
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    try:
        matches = re.search(r'[ABCDE]', s)
        if matches is None:
            for choice in choices:
                if s.lower() in choice.lower():
                    return choice[1]
            return ''
    except:
        return ''
    return matches[0]

def eval_text(task, response_text, last_gpt_answer_text, dialogue=None, other_info=None):
    if task=="vt":
        case_prompt = f"GPT text:{response_text}"
        prompt = TERMINATE + case_prompt
    elif task == "vi":
        case_prompt = f"GPT text:{response_text}"
        prompt = INTERRUPT + case_prompt
    elif task == "vw":
        case_prompt = f"GPT text:{response_text}"
        prompt = WAKE_UP + case_prompt
    elif task == "aw":
        case_prompt = f"GPT text:{response_text}\n\nGround Truth:{last_gpt_answer_text}"
        prompt = ANOMALY + case_prompt
    elif task =="hr":
        case_prompt = f"GPT text: {response_text}\n\nGround Truth:{last_gpt_answer_text}"
        prompt = HUMOR + case_prompt
    elif task =="gu":
        dialog_history = ""
        for i in dialogue[0:2]:
            
            dialog_history += f"From {i['from']}: {i['value']}\n"

        case_prompt = f"[Dialogue History]:\n{dialog_history}\n[Gesture]:{other_info['gesture']}\n\n[Reference Text]:{last_gpt_answer_text}\n\n Here is the GPT response to evaluate: {response_text}\n\n"
        prompt = GESTURE + case_prompt

    # print(prompt)
    
    result = gpt_score(prompt)

    return result


def count(args):
    
    src = args.src
    model = args.model
    task = args.task


    with open(src, 'r') as file:
        data = json.load(file)

    stats = defaultdict(lambda: defaultdict(int))
    eval_log = {}
    total = 0
    task_config = EVAL_CONFIG[task]

    with open(os.path.join(args.data_root, task_config["anno_file"])) as file:
        gts = json.load(file)
    
    gts_dict = { gt["video"]: gt for gt in gts}
    assert len(gts_dict) == len(gts)


    if task in ["aw","gu","vi","vt", "vw", "hr"]:
        for entry in tqdm(data):
            # print(entry)
            video = list(entry.keys())[0] #["video"]
            
            response_text = entry[video][0]
            gt = gts_dict[video]
            last_gpt_answer = gt["conversations"][-1]
            last_gpt_answer_text = gt["conversations"][-1]['value']
                            
         
          
            try:
                response_time = entry[video][1]
                if "timespan" in last_gpt_answer:
                    ref_time = last_gpt_answer["timespan"]
                else:
                    ref_time = last_gpt_answer["time"]
            
                eval_log[video] = {"ref_time":ref_time, "response_time":response_time}

                if in_time(response_time, ref_time):
                    stats[f'{task}']["time_correct"] += 1
                    stats[f'{task}']["has_answer_total"] += 1

                    if response_text == "":
                        response_text = "null"

                    eval_result = eval_text(task, response_text=response_text, last_gpt_answer_text=last_gpt_answer_text, dialog=gt["conversations"], other_info=gt)

                    if task == "gu":
                        score = float(eval_result['gesture']) + float(eval_result['context'])
                    elif task == "aw":
                        score = float(eval_result['description']) + float(eval_result['advice'])
                    else:
                        score = float(eval_result['score'])

                    stats[f'{task}']["answer_score"] += score
                    eval_log[video].update({"responsed text":response_text, "Ground truth text":last_gpt_answer_text,  "score":score, "eval_result": eval_result})
                    
            except Exception as e:
                print(e)
            
            total += 1
            stats[f'{task}']["total"] += 1

    elif task == "vr":
        for entry in tqdm(data):
            video = list(entry.keys())[0] #["video"]
            response_text = entry[video][0]
            gt = gts_dict[video]
            gpt_answer = gt["ans"]
            if extract_characters_regex(response_text)==gpt_answer:
            
                stats[f'{task}']["answer_correct"] += 1

            total += 1
            stats[f'{task}']["total"] += 1

    for task, counts in stats.items():
        task_config = EVAL_CONFIG[task]
        if task_config["time_acc"]:
            counts["time_accuracy"] = counts["time_correct"] / counts["total"] if counts["total"] > 0 else 0

        if task_config["text_score"]:
            counts["answer_average_score_overall"] = counts["answer_score"] / counts["total"] if counts["total"] > 0 else 0
            counts["answer_average_score_time_correct"] = counts["answer_score"] / counts["has_answer_total"] if counts["has_answer_total"] > 0 else 0
        else:
            counts["answer_correct"] = counts["answer_correct"] / counts["total"] *5 if counts["total"] > 0 else 0

    print(stats)
    os.makedirs("eval_results", exist_ok=True)
    # Save results as a JSON file
    with open(f'eval_results/{model}_{task}_stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)
    with open(f'eval_results/{model}_{task}_log.json', 'w') as json_file:
        json.dump(eval_log, json_file, indent=4)

    print(f"{total} items have been statisticed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--src', type=str, help='Path to the data file')
    parser.add_argument('--task', type=str, help='Task name')
    parser.add_argument('--data-root', type=str, default="ViSpeak-Bench")
    args = parser.parse_args()
    count(args)

if __name__ == "__main__":
    main()
