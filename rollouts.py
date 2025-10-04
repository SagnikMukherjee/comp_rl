import json # Added missing import
# Import vLLM libraries
from vllm import LLM, SamplingParams
import torch # Added missing import
from transformers import AutoTokenizer
import re
import argparse
from tqdm import tqdm
from utils import parse_final_answer

INSTRUCTIONS = {
    "PRIME-RL/Eurus-2-7B-SFT": "\n\nPresent the answer in LaTex format: \\boxed{Your answer}",
    "deepseek-ai/deepseek-math-7b-instruct": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "Qwen/Qwen2.5-7B-Instruct": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "Qwen/Qwen2.5-7B": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "Qwen/Qwen3-1.7B-Base": "Let's think step by step and output the final answer within \\boxed{}."
}

def load_prompts_from_file(filepath):
    with open(filepath, 'r') as f:
        prompts = json.load(f)
    return prompts



def generate_rollouts(prompt_file, model_string_name, num_rollouts_per_prompt, rollout_output_file, num_samples=None):
    prompts = load_prompts_from_file(prompt_file)

    tokenizer = AutoTokenizer.from_pretrained(model_string_name)
    if num_samples:
        print(num_samples)
        prompts = prompts[:num_samples]
    
    llm = LLM(model=model_string_name, tensor_parallel_size=4, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=1, top_p=1, top_k=-1, max_tokens=1024, n=num_rollouts_per_prompt)

    rollout_results_to_dump = []
    instruction = INSTRUCTIONS[model_string_name]


    for i, prompt in enumerate(tqdm(prompts)):
        print(prompt)
        try:
            question, solution, answer = prompt["problem"], prompt["solution"], prompt["answer"]
        except:
            question, solution, answer = prompt["problem"], prompt["solution"], str(parse_final_answer(prompt["solution"]))
        messages = [
            {"role": "user", "content": question + " " + instruction}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)

        rollouts_details = []
        correct_count = 0
        for output in outputs:
            for gen_output in output.outputs:
                generated_text = gen_output.text
                parsed_answer = str(parse_final_answer(generated_text))
                rollouts_details.append({"generated_text": generated_text, "final_answer": parsed_answer})
                
                try:
                    if parsed_answer == answer:
                        correct_count += 1
                except Exception as e:
                    print(e)
        
        prompt_accuracy = correct_count / len(rollouts_details) if rollouts_details else 0
        rollout_results_to_dump.append({
            "question": question, 
            "solution": solution,
            "answer": answer,
            "rollouts": rollouts_details,
            "accuracy": prompt_accuracy
        })
        if(i%10==0):
            with open(rollout_output_file, "w") as f:
                json.dump(rollout_results_to_dump, f, indent=4)


    with open(rollout_output_file, "w") as f:
        json.dump(rollout_results_to_dump, f, indent=4)

    del llm # Clear vLLM model from GPU memory
    torch.cuda.empty_cache()

    return prompts, rollout_results_to_dump


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_file", type=str, required=True)
    argparser.add_argument("--model_name", type=str, required=True)
    argparser.add_argument("--num_rollouts_per_prompt", type=int, required=True)
    argparser.add_argument("--rollout_output_file", type=str, required=True)
    argparser.add_argument("--num_samples", type=int, required=False, default=None)
    args = argparser.parse_args()
    
    generate_rollouts(args.prompt_file, args.model_name, args.num_rollouts_per_prompt, args.rollout_output_file, args.num_samples)
    