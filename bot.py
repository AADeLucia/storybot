#!/Users/alexandradelucia/anaconda3/envs/redditbot/bin/python

"""
"""
# Standard
import logging
import json
import re
import sys
import time
import queue

# Third-party
import praw
from praw.exceptions import RedditAPIException
from prawcore.exceptions import Forbidden, OAuthException
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load settings
with open("config.json", "r") as f:
    config = json.loads(f.read())

# Load model
model_path = config["model_path"]
stop_token = "<|endoftext|>"
max_length = 1000
device = "cpu"
torch.device(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
logging.info(f"\n\n")


def get_model_response(prompt):
    """Get response from the model"""
    # Encode prompt
    full_prompt = f"{prompt} [RESPONSE]"
    logging.debug(f"prompt: {full_prompt}")
    encoded_prompt = tokenizer.encode(full_prompt, return_tensors="pt")
    logging.debug(f"encoded prompt: {len(encoded_prompt)} {encoded_prompt}")
    encoded_prompt = encoded_prompt.to(device)

    # Generate
    output_sequence = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length + len(encoded_prompt[0]),
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        pad_token_id=50256,
        repetition_penalty=1.0,
        do_sample=True,
        num_beams=1,
        num_return_sequences=1,
    )
    output_sequence = output_sequence[0]
    logging.debug(f"output_sequence: {output_sequence}")

    # Decode response
    response = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
    response = re.split("(\[RESPONSE\])", response)[-1]
    response = re.split("(<\|endoftext\|>)", response)[0]
    response = re.sub("<newline>", "\n", response)
    return response


def format_reply(response):
    return f"{response}\n\n----------------\nAutomatically generated response using GPT-2. See the [StoryBot GitHub](https://github.com/AADeLucia/storybot) for details."


if __name__ == "__main__":
    try:
        # Reddit service
        # Initialize Reddit service
        reddit = praw.Reddit(
            user_agent=config["user_agent"],
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            username=config["username"],
            password=config["password"]
        )

        # Test credentials
        reddit.user.me()
    except OAuthException as err:
        logging.error(f"Issue connecting to Reddit. Check login values.\n{err}")
        sys.exit(1)

    # Search r/WritingPrompts
    # Want the top 10 posts of the day
    subreddit = reddit.subreddit("WritingPrompts")
    for sub in subreddit.search("flair:WritingPrompt", time_filter="day", sort="new", limit=5):
        try: 
            response = get_model_response(sub.title)
            logging.info(f"{sub.title}\n{response}\n\n")
            sub.reply(format_reply(response))
        except RedditAPIException as err:
            logging.warning(f"Hit rate limit. Sleeping and trying again.")
            time.sleep(60)
            continue


