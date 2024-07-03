import logging, sys

from huggingface_hub import snapshot_download
from tenacity import retry, wait_random_exponential, before_sleep_log
import json

from tqdm import tqdm

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

with open('splits_full.json', 'r') as f:
    data = json.load(f)
demo_names = []

for k,v in data.items():
    demo_names += data[k]

demo_names = set(demo_names)
try:
    with open('completed.txt', 'r') as f:
        for line in f:
            demo_names.remove(line.strip())
except FileNotFoundError:
    print("Completed file not found")

@retry(wait=wait_random_exponential(multiplier=1), before_sleep=before_sleep_log(logger, logging.INFO))
def download_demonstrations_with_retry(pattern):
    snapshot_download(
        repo_id="McGill-NLP/WebLINX-full",
        repo_type="dataset",
        local_dir="ENTER LOCAL DIR HERE",
        allow_patterns=[pattern]
    )


for name in (pbar := tqdm(demo_names)):
    pbar.set_description(f"Downloading {name}")
    pattern = f"demonstrations/{name}/*"
    download_demonstrations_with_retry(pattern)
    with open('completed.txt', 'a') as f:
        f.write(f"{name}\n")
print("completed")
