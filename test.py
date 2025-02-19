from openai import OpenAI
import re

client = OpenAI()

max_iterations = 8
criteria = ["relevance", "coherence", "simplicity", "depth"]

def score_prompt(prompt):
    eval_format = "\n".join([f"- {criterion.capitalize()}: [score]" for criterion in criteria])
    instructions = f"""
    Evalue the following prompt based on the criteria {', '.join(criteria)}.
    Provide a score for each factor on a scale from 1 to 10 and calculate a final average.

    Prompt: "{prompt}"

    Your response should be in this format:
    {eval_format}
    - Average Score: [average]
    """

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role" : "system", "content" : "You are an AI evaluator tasked with scoring prompts based on certain criteria"},
            {"role" : "user", "content" : instructions}
        ]
    )
    return response.choices[0].message.content

def parse_scores(text):
    scores = {}
    for criterion in criteria:
        match = re.search(fr"{criterion.capitalize()}: (\d+)", text)
        if match:
            scores[criterion] = int(match.group(1))
    
    if scores:
        scores["average"] = sum(scores.values()) / len(scores)
    else:
        scores["average"] = None  
    
    return scores
    

def generate_response(prompt, criteria):
    criteria_text = f"{', '.join(criteria)}"
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role" : "system", "content" : "You are an AI that elps improve the specifity and clarity of prompts."},
            {"role" : "user", "content" : f"Please refine this prompt: {prompt}, focusing on improving these elements: {criteria_text}"}
        ]
    )
    return response.choices[0].message.content

def find_improvement(d1, d2):
    res = []
    for criterion in d1:
        if criterion == "average":
            return res
        old_score, new_score = d1[criterion], d2[criterion]
        if old_score > new_score:
            res.append(criterion)
            

def main():
    initial_prompt = input("Give me a prompt to improve: ")
    initial_scores = parse_scores(score_prompt(initial_prompt))
    
    prompt = generate_response(initial_prompt, criteria)
    scores = parse_scores(score_prompt(prompt))
    to_improve = find_improvement(initial_scores, scores)

    total_iters = 0
    cur_iters = 0
    while cur_iters < 2 or total_iters < max_iterations:
        prompt = generate_response(prompt, to_improve)
        cur_scores = parse_scores(score_prompt(prompt))
        to_improve = find_improvement(scores, cur_scores)
        
        if len(to_improve) == 0:
            cur_iters += 1
        
        total_iters += 1
    
    print(f"Final Prompt: {prompt}")

if __name__ == '__main__':
    main()