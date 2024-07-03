import logging, sys

from huggingface_hub import snapshot_download
from tenacity import retry, wait_random_exponential, before_sleep_log

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(wait=wait_random_exponential(multiplier=1), before_sleep=before_sleep_log(logger, logging.INFO))
def download_with_retry():
    snapshot_download(
        repo_id="McGill-NLP/WebLINX-full",
        repo_type="dataset",
        local_dir="ENTER LOCAL DIR HERE",
        resume_download=True,
        ignore_patterns=["demonstrations/*"]
    )

download_with_retry()
