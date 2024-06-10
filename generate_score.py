from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", trust_remote_code=True, token="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")

from llama_cpp import Llama
model = Llama(
    model_path='models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf',
    n_gpu_layers=-1,
    n_ctx=8192,
)

from datasets import load_dataset
dataset = load_dataset("json", data_files="filtered_oscar.jsonl", split="train")

def generate_prompt(text):
    prompt = f'''Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.
- Evaluation should be based on Japan.

The extract:
{text}

After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score:  <total points>"'''
    conversations = [
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

    return formatted_prompt

def get_score(prompt, output):
    output_text = output["choices"][0]["text"][len(prompt):]
    try:
        reason = output_text.split("Educational score: ")[0]
        score = int(output_text.split("Educational score: ")[1][0])
        return reason, score
    except:
      return output_text, None

import jsonlines
from tqdm import tqdm

count = 0
# OSCARデータのフィルタリングとJSONLファイルへの保存
with jsonlines.open('oscar_mixtral_scored_v2.jsonl', mode='w') as writer:
    for item in tqdm(dataset, desc="Processing texts"):
        try:
            text = item['text']
            if len(text) > 5000:
                text = text[:5000]
            prompt = generate_prompt(text)
            output = model(prompt, max_tokens=256,stop=["</s>", "[INST]"],echo=True)
            reason, score = get_score(prompt, output)
            if score is not None:
                print(f"Text: {text[:100]}")
                print(f"Reason: {reason}")
                print(f"Score: {score}\n")
                writer.write({'id': count, 'score': score, 'reason': reason, 'text': text})
                count += 1
        except Exception as e:
            print(f"Error processing text: {text}")
            print(e)
            count += 1
