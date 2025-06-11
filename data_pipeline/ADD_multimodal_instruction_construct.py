import json
import os
import cv2
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import requests
import openai

# Helper function to encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to convert mask to bounding box
def mask_to_bbox(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask image from {mask_path}")

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

# Function to call GPT-4o API
def chat(prompt, base64_image):
    base_url = 
    api_version = 
    ak = 
    model_name = 
    max_tokens = 100

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are good at describing objects in an image based on bounding box and context.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            extra_headers={"X-TT-LOGID": "${your_logid}"},
        )
        return completion.model_dump_json()
    except Exception as e:
        print(f"Error in API request: {e}")
        return None

# Function to process each data entry
def process_entry(entry):
    try:
        #mask_path = entry['mask_path'][0]
        x, y, w, h = entry['bbox'][0], entry['bbox'][1], entry['bbox'][2], entry['bbox'][3] #mask_to_bbox(mask_path)
        width, height = entry['width'], entry['height']
        object_name = entry['object']
        image_path = entry['image']

        base64_image = encode_image(image_path)
        prompt = (
            f"The object is '{object_name}'. It is located in a bounding box with coordinates ({x}, {y}, {w}, {h}) "
            f"on an image of size {width}x{height}. Describe its size, relative position, and relation to surrounding objects. Avoid describing the overall scene or unrelated elements.\n "
            f"Your response should start with 'Add <imagehere> to the [position] of <imagehere>' (The first <imagehere> indicates the object and the second indicates the image.) Keep the <imagehere> symbol in the first sentence in your reply.\n "
            f"Answer briefly in two sentences: "
        )

        response = chat(prompt, base64_image)

        if response:
            entry['multi_prompt'] = json.loads(response)['choices'][0]['message']['content']
            return entry
        
    except Exception as e:
        print(f"Error processing entry {entry.get('idx', 'unknown')}: {e}")
        return None

# Main function to process the file
def process_file(file_path, output_path, max_workers=5, timeout=10):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_entry, entry): entry for entry in data}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing entries"):
            try:
                result = future.result(timeout=timeout)  # Set timeout for each entry
                if result:
                    results.append(result)
            except TimeoutError:
                entry = futures[future]
                print(f"Timeout for entry idx: {entry.get('idx', 'unknown')}")
            except Exception as e:
                print(f"Error processing entry: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Example usage
input_file = path/to/input.json 
output_file = path/to/output.json'

process_file(input_file, output_file, max_workers=10, timeout=10)
