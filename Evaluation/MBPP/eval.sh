OUPUT_DIR="exp/rag"
MODEL="deepseek-coder-v2-236B-instruct"

python eval_instruct.py \
    --model "deepseek-ai/$MODEL" \
    --output_path "$OUPUT_DIR/${LANG}.$MODEL.jsonl" \
    --temp_dir $OUPUT_DIR \
    --use_rag