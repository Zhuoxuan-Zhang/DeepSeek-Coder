from human_eval.evaluation import evaluate_functional_correctness
from prompts.rag_prompt import construct_rag_prompt
import argparse
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="",
                base_url="https://api.deepseek.com")

data_abs_dir = Path(__file__).parent / "data"


def read_test_examples(data_path: str, use_rag=True):
    def format_test_example(q, tests, code: str = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(
            q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']

        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
'''.strip().format('\n\n'.join(examples_str), prompt)
        #NOTE: we give mbpp code directly
        if use_rag:
            prompt_with_shots += construct_rag_prompt(code)
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_with_shots
        }


def convert_for_evaluation(example):
    gpt_completion = example['gpt_completion']
    generation = gpt_completion
    try:
        code_block: str = re.findall(
            f'```python\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example['generation'] = generation
    return example


def generate_one(example):
    prompt = example['prompt']
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )

    output = response.choices[0].message.content

    # print(output)
    # exit()
    example['gpt_completion'] = output
    return convert_for_evaluation(example)


def generate_main(args):
    model_name_or_path = args.model
    saved_path = args.output_path
    temp_dir = args.temp_dir
    use_rag = args.use_rag
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"mbpp.jsonl")

    examples = list(read_test_examples(problem_file, use_rag=use_rag))
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []
    for ex in tqdm(examples, desc='Generating'):
        gen_example = generate_one(ex)
        generated_examples.append(gen_example)
        print("Generate {}/{} over...".format(len(generated_examples), len(examples)))

    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(
            len(generated_examples), saved_path))

    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        problem_file=os.path.join(data_abs_dir, f"mbpp_test.jsonl"),
        language='python',
        is_mbpp=True
    )
    print(result, model_name_or_path)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path")
    parser.add_argument('--output_path', type=str,
                        help="output path of your generation")
    parser.add_argument('--temp_dir', type=str,
                        help="temp dir for evaluation", default="tmp")
    parser.add_argument('--use_rag', action='store_true',
                        help="whether to use rag for generation")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass
