
DATA_ROOT=./data

MODEL_NAME=vispeak
RESULT=../results/vispeak

python count.py --model $MODEL_NAME --task "aw" --src "${RESULT}/Anomaly_Warning_output.json" --data-root $DATA_ROOT
python count.py --model $MODEL_NAME --task "gu" --src "${RESULT}/Gesture_Understanding_output.json" --data-root $DATA_ROOT
python count.py --model $MODEL_NAME --task "vi" --src "${RESULT}/Visual_Interruption_output.json" --data-root $DATA_ROOT
python count.py --model $MODEL_NAME --task "vt" --src "${RESULT}/Visual_Termination_output.json" --data-root $DATA_ROOT
python count.py --model $MODEL_NAME --task "vw" --src "${RESULT}/Visual_Wake-Up_output.json" --data-root $DATA_ROOT
python count.py --model $MODEL_NAME --task "hr" --src "${RESULT}/Humor_Reaction_output.json" --data-root $DATA_ROOT
python count.py --model $MODEL_NAME --task "vr" --src "${RESULT}/Visual_Reference_output.json" --data-root $DATA_ROOT


