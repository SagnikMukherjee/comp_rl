import json
import string
import random
import re

with open("/home/sagnikm3/comp_rl/data/math.json", "r") as f:
    data=json.load(f)

data_new = []
for i in range(len(data)//2):
    q1, q2= data[i*2], data[i*2+1]

    chosen = [q1,q2]
    var_names = list(string.ascii_lowercase[:len(chosen)])
    subqs, sources, values, source_answers = [], [], {}, []
    for i, p in enumerate(chosen):
        q, a = p["problem"], p["answer"]
        var=var_names[i]
        phr = random.choice([
            f"Sub-question {i+1}: {q} Call this result {var}.",
            f"Sub-question {i+1}: {q} Let the answer be {var}.",
            f"Sub-question {i+1}: {q} Denote this value as {var}."
        ])

        subqs.append(phr)
        sources.append(q)
        source_answers.append(a)


        values[var] = float(a)

    ops = [
        ("a+b", lambda a,b: (a+b)),
    ]

    available_vars = set(values.keys())
    candidate_ops = []
    for op_expr, fn in ops:
        needed_vars = set(re.findall(r"[a-z]", op_expr))
        candidate_ops.append((op_expr, fn, needed_vars))
    
    op, fn, vars_needed = random.choice(candidate_ops)
    result = fn(**{v: values[v] for v in vars_needed})
    final_q = f"Final question: Compute {op.replace('*','×').replace('/','÷').replace('^','^')}."
    question = "\n".join(subqs+[final_q])
    data_new.append(
        {
            "problem": question,
            "solution": f"the answer is $\\boxed{{{result}}}$"
        }
    )


with open("/home/sagnikm3/comp_rl/data/math_perturbed.json", "w") as f:
    json.dump(data_new, f)

