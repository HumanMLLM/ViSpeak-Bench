EVAL_MODEL="VITA1P5"
Devices=0
DATA_ROOT=./data
export PYTHONPATH=./:../VITA


TASK="Visual_Reference"
data-file="../data/annotations/Visual_Reference.json"
output-file="../results/${EVAL_MODEL}/${TASK}_output.json"
BENCHMARK="Vispeak"
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model-name $EVAL_MODEL --benchmark_name $BENCHMARK --data-file $data-file --output-file $output-file --video-root $DATA_ROOT



TASK="Gesture_Understanding"
data-file="../data/annotations/Gesture_Understanding.json"
output-file="../results/${EVAL_MODEL}/${TASK}_output.json"
BENCHMARK="VispeakProactive"
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model-name $EVAL_MODEL --benchmark_name $BENCHMARK --data-file $data-file --output-file $output-file --video-root $DATA_ROOT


TASK="Anomaly_Warning"
data-file="../data/annotations/Anomaly_Warning.json"
output-file="../results/${EVAL_MODEL}/${TASK}_output.json"
BENCHMARK="VispeakProactive"
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model-name $EVAL_MODEL --benchmark_name $BENCHMARK --data-file $data-file --output-file $output-file --video-root $DATA_ROOT


TASK="Humor_Reaction"
data-file="../data/annotations/Humor_Reaction.json"
output-file="../results/${EVAL_MODEL}/${TASK}_output.json"
BENCHMARK="VispeakProactive"
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model-name $EVAL_MODEL --benchmark_name $BENCHMARK --data-file $data-file --output-file $output-file --video-root $DATA_ROOT


TASK="Visual_Interruption"
data-file="../data/annotations/Visual_Interruption.json"
output-file="../results/${EVAL_MODEL}/${TASK}_output.json"
BENCHMARK="VispeakProactive"
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model-name $EVAL_MODEL --benchmark_name $BENCHMARK --data-file $data-file --output-file $output-file --video-root $DATA_ROOT



TASK="Visual_Termination"
data-file="../data/annotations/Visual_Termination.json"
output-file="../results/${EVAL_MODEL}/${TASK}_output.json"
BENCHMARK="VispeakProactive"
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model-name $EVAL_MODEL --benchmark_name $BENCHMARK --data-file $data-file --output-file $output-file --video-root $DATA_ROOT


TASK="Visual_Wake-Up"
data-file="../data/annotations/Visual_Wake-Up.json"
output-file="../results/${EVAL_MODEL}/${TASK}_output.json"
BENCHMARK="VispeakProactive"
CUDA_VISIBLE_DEVICES=$Devices python eval.py --model-name $EVAL_MODEL --benchmark_name $BENCHMARK --data-file $data-file --output-file $output-file --video-root $DATA_ROOT
 

