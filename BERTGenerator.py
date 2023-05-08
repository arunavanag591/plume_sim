import pandas as pd
import numpy as np
import re
import concurrent.futures
from transformers import BertTokenizer, BertForMaskedLM
import torch


# Load and preprocess data
dir = '~/DataAnalysis/data/Sprints/HighRes/'
df = pd.read_hdf(dir+'Windy/WindyMASigned.h5')
df1=df[0:150000]
df1=df1.round(3)

# Test Dataset
test=df[15000:25000]
test=test.round(3)


# Set the chunk size
chunk_size = 60

# Divide the DataFrame into smaller chunks
num_chunks = (len(df1) - 1) // chunk_size + 1

for chunk_index in range(num_chunks):
    start_index = chunk_index * chunk_size
    end_index = min((chunk_index + 1) * chunk_size, len(df1))
    chunk_df = df1.iloc[start_index:end_index]

    # Convert the chunk DataFrame to textual format
    text_data = ""
    for idx, row in chunk_df.iterrows():
        text_data += f"Odor encounter: {row['odor']}, Location: ({row['xsrc']}, {row['ysrc']}), U velocity: {row['U']}, V velocity: {row['V']}.\n"


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()


def generate_odor_with_bert(input_row):
    prompt = f"{text_data}\nGenerate a new odor encounter with the following location: ({input_row['xsrc']}, {input_row['ysrc']}) and U velocity: {input_row['U']}, V velocity: {input_row['V']}.Please use the format: Odor encounter: X, Location: (X, Y)"

    pattern = r"Odor encounter: ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?), Location: \(([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?), ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\)"
    match = None

    while not match:
        # Tokenize the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt',max_length=512, truncation=True)
        mask_idx = len(input_ids[0]) - 1

        # Mask the last token (X)
        input_ids[0, mask_idx] = tokenizer.mask_token_id

        # Generate logits for masked token
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Sample the top 10 predicted tokens
        top_k = 10
        probs = torch.nn.functional.softmax(logits[0, mask_idx], dim=-1)
        top_k_indices = torch.topk(probs, top_k).indices

        # Generate text for each predicted token
        for index in top_k_indices:
            generated_text = tokenizer.decode(input_ids[0].tolist()[:-1] + [index.item()])
            match = re.search(pattern, generated_text)
            if match:
                break

    if match:
        new_entry = {
            "odor": float(match.group(1).strip()),
            "xsrc": float(match.group(2).strip()),
            "ysrc": float(match.group(3).strip()),
        }
        return new_entry
    else:
        print("Failed to parse generated text")
        print("Generated text:", generated_text)
        return None


def main():
 
    # Create a new DataFrame to store generated data
    generated_df = pd.DataFrame(columns=['odor', 'xsrc', 'ysrc'])
    num_threads = 8 

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(generate_odor_with_bert, test.to_dict('records')))

    for result in results:
        if result is not None:
            generated_df = pd.concat([generated_df, pd.DataFrame([result])], ignore_index=True)

    generated_df.to_hdf('../generated.h5',key='generated_df',mode='w')



if __name__ == "__main__":
    main()