import os
import random
from datasets import load_dataset
from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.play import save
from tqdm import tqdm


load_dotenv()

client = SarvamAI(api_subscription_key=os.getenv('SARVAMAI_API_KEY'))

dataset = load_dataset("santhosh/english-malayalam-names", split="train", streaming=False)
df = dataset.to_pandas()

speakers = ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"]

for i, text in tqdm(enumerate(df["ml"].iloc[:1000])):
    folder = f"output"
    os.makedirs(folder, exist_ok=True)
    text = f"എന്റെ പേര് {text}." 

    try:
        audio = client.text_to_speech.convert(
            target_language_code="ml-IN",
            text=text,
            model="bulbul:v2",
            speaker=random.choice(speakers)
        )
        save(audio, f"{folder}/{i}_mlnames.wav")
    except Exception as e:
        print(f"Error at index {i}: {e}")
