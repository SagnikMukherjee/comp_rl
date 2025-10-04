import argparse
from datasets import load_dataset
import json
import re
from utils import remove_boxed, last_boxed_only_string
from tqdm import tqdm
from sympy.parsing.latex import parse_latex
from sympy import SympifyError

def parse_answer(latex_str):
    try:
        return parse_latex(latex_str)   # SymPy object (Rational, Symbol, etc.)
    except Exception:
        return latex_str  # fallback to raw string if parsing fails

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        data = [{"question": ex["question"], "solution": ex["answer"], "answer": re.sub(r'[,]', '', ex["answer"].split("####")[-1].strip())} for ex in ds["train"]]
        with open(args.output_dir + "/gsm8k.json", "w") as f:
            json.dump(data, f, indent=4)

    elif args.dataset == "math":
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", trust_remote_code=True)
        
        data = []
        for ex in tqdm(ds["train"], desc="Processing"):
            try:
                ans = float(parse_answer(remove_boxed(last_boxed_only_string(ex["solution"]))))
            except Exception as e:
                print(e)
                continue
            data.append({
                "question": ex["problem"],
                "solution": ex["solution"],
                "answer": ans
            })
        with open(args.output_dir + "/math.json", "w") as f:
            json.dump(data, f, indent=4)
    

    elif args.dataset == "dapomath":
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", trust_remote_code=True)
        

        data = []
        for ex in tqdm(ds["train"], desc="Processing examples"):
            item = {
                "question": ex["prompt"][0]["content"].replace(
                    '\n\nRemember to put your answer on its own line after "Answer:".', ""
                ).replace("Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n", ""),
                "solution": " ",
                "answer": ex["reward_model"]["ground_truth"],
            }
            data.append(item)
        with open(args.output_dir + "/dapomath.json", "w") as f:
            json.dump(data, f, indent=4)
    
if __name__ == "__main__":
    main()