import asyncio
import json
import httpx
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef

API_KEY = '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'
# Set up the request parameters
url = "https://api.openai.com/v1/chat/completions"
# Set up the request headers
headers = {
    "Content-Type": "application/json",
    # Replace API_KEY with your OpenAI API key
    "Authorization": f"Bearer {API_KEY}"
}


with open("/scratch/ace14856qn/mimic/test_mimic.json") as f:
    test_data = f.readlines()
test_data = [json.loads(val) for val in test_data]
with open("/scratch/ace14856qn/mimic/valid_mimic.json") as f:
    valid_data = f.readlines()
valid_data = [json.loads(val) for val in valid_data]
print (test_data[0], valid_data[0])

lock = asyncio.Lock()

predictions = []

role_prompt = "You are a chest radiologist that identify the main findings and diagnosis or impression based on the given FINDINGS section of the chest X-ray report, which details the radiologists’ assessment of the chest X-ray image. Please ensure that your response is concise and does not exceed the length of the FINDINGS."
instruction_prompt = "Please ensure that your response is concise and does not exceed the length of the FINDINGS."

sample_prompt = "What are the main findings and diagnosis or impression based on the given Finding in chest X-ray report? FINDINGS:"
output_prompt = "IMPRESSION:"

def build_sample(sample, test=False):
    text = f"{sample_prompt} {sample['findings']}"
    if test:
        text = f"{text}\n{output_prompt} {sample['impression']}"
    return text

valid_samples = "\n".join([build_sample(val, test=True) for val in valid_data])
predictions = []

async def process_row(sample, valid_samples=None, progress_bar=None, client=None):
    prompt = f"{instruction_prompt}"
    if valid_samples:
        prompt = f"{prompt}\n{valid_samples}"
    prompt = f"{prompt}\n{build_sample(sample)}"
    # Set up the request data
    data = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 120,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }
    print (prompt)

    while True:
        try:
            r = await client.post(url, json=data, headers=headers,
                    timeout=10)
            json_response = r.json()
            print (json_response)
            content = json_response['choices'][0]['message']['content']
            break
        except Exception as e:
            raise e
            continue

    async with lock:
        print (content)
        predictions.append({"pred": content, "label": sample["impression"]})
        progress_bar.update(1)
# await asyncio.sleep(5)  # Wait for seconds before making the next request


def evaluate(y_true, y_pred):
    return accuracy_score(y_true, y_pred), matthews_corrcoef(mcc_trans(y_true),
            mcc_trans(y_pred))

async def main():
    with tqdm(total=len(test_data)) as progress_bar:
        async with httpx.AsyncClient() as client:
            tasks = [process_row(test_data[index], progress_bar=progress_bar, client=client)
                    for index in range(len(test_data))][:1]
            await asyncio.gather(*tasks)

    # print (evaluate(tes_gt[:len(predictions)], predictions))

if __name__ == '__main__':
    asyncio.run(main())
