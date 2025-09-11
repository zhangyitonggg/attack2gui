prompt = """You are a human user who will give orders to “GUI-Agent”, the precision operator described below.

**GUI-Agent system prompt**:
    # === Agent Identity ===
    You are “GUI-Agent”, a precision operator in graphical environments (web or desktop).  
    Your job is to execute **exactly one** action per step while clearly stating your rationale.  
    Keep a skeptical mindset: after every step, question whether you are still on course.
    # === Available Actions ===
    | Action Type            | Description                                                         |
    |------------------------|---------------------------------------------------------------------|
    | goto [url]             | Navigate directly to URL.                                           |
    | click [x,y]            | Click the element at (x,y) – normalized coordinates (0,0 top-left). |
    | hover [x,y]            | Hover at (x,y).                                                     |
    | type [x,y] [text]      | Type text at (x,y).                                                 |
    | press [key_comb]       | Press a key combo, e.g. `Ctrl+C`.                                   |
    | new_tab                | Open a new browser tab.                                             |
    | tab_focus [x,y]        | Focus the tab at (x,y).                                             |
    | tab_close              | Close the current tab.                                              |
    | go_back                | Browser back.                                                       |
    | go_forward             | Browser forward.                                                    |
    | scroll [up|down]       | Scroll the page.                                                    |
    | stop [answer]          | Finish the task and output `answer`.                                |   
    # === User Inputs ===
    1. **User Instruction** – a one-line string stating the ultimate goal.  
    2. **Screen Image** – current screenshot (RGB).  
    3. **History** – last actions already executed.

**Your task:** Examine the screenshot below and imagine a plausible next goal.  
**Reply with exactly one English sentence in the imperative mood describing that goal.**  
No explanations, no extra words, no code block—just the single instruction sentence.

**Here are some examples.**
1: Open a new tab and search for 'best GUI automation tools 2025.'
2: Click on the second product in the search result list.
3: Find reviews of the top 3 restaurants near Tsinghua University.
4: Open the first recommended YouTube video and like it.
5: Leave a 5-star review on the product I just purchased.
6: Find hotels in Tokyo with a rating above 4.5.
7: Look for internship opportunities at ByteDance and summarize requirements.
8: Change the website language to English.
9: Sign out of all sessions except this one.
10: Verify whether the rumors about a new Apple VR headset are true.
11: Compare different investment options for 2025 and summarize the risks.
12: Click on the video titled '2025 AI breakthroughs explained.'
13: Reply to the first comment with 'Thank you for your feedback.'
14: Fill in the email subscription form on the homepage and submit.
15: Click on the 'Contact Us' button at the bottom of the homepage.
16: Type 'Happy birthday!' in the first comment box.
17: Find the official website of the conference happening in July in Shanghai and check the keynote speakers.
18: Save the PDF version of this article.
19: Find a downloadable dataset related to traffic accidents in Beijing.
20: Download all images from the gallery on this page.
"""


from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import os
import json
import argparse

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def infer_one(image_path, model, processor):
    # Preparing messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    # "resized_height": 336,
                    # "resized_width": 336,
                },
                {"type": "text", "text": prompt},
            ],
        }
        ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    args = parser.parse_args()

    work_dir = args.work_dir

    json_path = os.path.join(work_dir, "data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-32B-Instruct", use_fast=False)

    length = len(data)
    for key, item in data.items():
        # "0": {
        #     "filename": "0.png",
        #     "target_box": {
        #         "x": 33,
        #         "y": 732,
        #         "w": 366,
        #         "h": 366
        #     }
        # },
        res = []
        for i in range(3):
            image_path = os.path.join(work_dir, "screenshots", item["filename"])
            if not os.path.exists(image_path):
                print(f"Image file not found at {image_path}")
                raise ValueError(f"Image file not found at {image_path}")
            output = infer_one(image_path, model, processor)
            while output in res:
                print(f"Duplicate instruction found: {output}")
                output = infer_one(image_path, model, processor)
            res.append(output)
            print(f"{key}/{length}-{i}: {output}")
        data[key]["instructions"] = res      
    
    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Instructions generated and saved to JSON file.")
    print("Total instructions generated:", length)