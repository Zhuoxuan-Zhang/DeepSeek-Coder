OUPUT_DIR="output"
MODEL="deepseek-coder-33b-instruct"

python eval_instruct.py \
    --model "deepseek-ai/$MODEL" \
    --output_path "$OUPUT_DIR/${LANG}.$MODEL.jsonl" \
    --temp_dir $OUPUT_DIR