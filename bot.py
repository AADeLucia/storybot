#!/Users/alexandradelucia/anaconda3/envs/redditbot/bin/python

"""
"""
# Standard
import logging
import json
import re
import sys
import time
import argparse

# Third-party
import praw
from praw.exceptions import RedditAPIException
from prawcore.exceptions import Forbidden, OAuthException
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


###
# Model settings
###
stop_token = "<|endoftext|>"
MAX_LENGTH = 1000
device = "cpu"
torch.device(device)


def reddit_login(config):
    """Login to Reddit"""
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


def _generate(encoded_prompt, max_length):
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
    return output_sequence


def get_model_response(prompt):
    """Get response from the model"""
    # Encode prompt
    full_prompt = f"{prompt} [RESPONSE]"
    logging.debug(f"prompt: {full_prompt}")
    encoded_prompt = tokenizer.encode(full_prompt, return_tensors="pt")
    logging.debug(f"encoded prompt: {len(encoded_prompt)} {encoded_prompt}")
    encoded_prompt = encoded_prompt.to(device)

    # Generate response
    max_length = MAX_LENGTH
    while True:
        try:
            output_sequence = _generate(encoded_prompt, max_length=max_length)
            break
        except IndexError as err:
            logging.warning(f"Max length {max_length} caused error. Reducing length by 50:\n {err}")
            max_length -= 50

    # Decode response
    response = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
    response = re.split("(\[RESPONSE\])", response)[-1]
    response = re.split("(<\|endoftext\|>)", response)[0]
    response = re.sub("<newline>", "\n", response)

    # Sanitize quotes and horizontal line symbols for Reddit
    response = re.sub("``(\s)*", "\"", response)
    response = re.sub("(\s)*''", "\"", response)
    response = re.sub("[~\*\-]{3,}", "\n", response)

    return response


def format_reply(response):
    return f"{response}\n\n----------------\nAutomatically generated response using GPT-2. See the [StoryBot GitHub](https://github.com/AADeLucia/storybot) for details."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, help="Path to log file for output. Default is STDOUT")
    parser.add_argument("--wait-time", type=int, default=600, help="Wait time between failed API calls (seconds)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Direct output to logfile
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load settings
    with open("config.json", "r") as f:
        config = json.loads(f.read())

    # Login to Reddit
    reddit = reddit_login(config)

    # Load model
    model_path = config["model_path"]
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    logging.info(f"\n\n")

    # Search r/WritingPrompts
    # Want the top 10 posts of the day
    # Cross-post them to r/StoryGenAI and reply to them
    subreddit = reddit.subreddit("WritingPrompts")
    logging.info(reddit.auth.limits)

    # Get top 10 WritingPrompt posts of the day
    top_writing_prompts = subreddit.search("flair:WritingPrompt", time_filter="day", sort="top", limit=10)

    for post in top_writing_prompts:
        logging.info(reddit.auth.limits)

        # Irritating while loops to combat rate limit
        while True:
            try: 
                # Cross-post
                crosspost_sub = post.crosspost("StoryGenAI", send_replies=False, flair_id="79f236dc-a93b-11ea-ab34-0e2ac1b274db")
                break
            except RedditAPIException as err:
                logging.warning(f"Hit rate limit. Sleeping and trying again.\n{err}")
                time.sleep(args.wait_time)

        # Generate a response to the post
        response = get_model_response(post.title)
        logging.info(f"{post.title}\n{response}\n\n")
        
        # Reply to the new post
        while True:
            try: 
                reply = crosspost_sub.reply(format_reply(response))
                break
            except RedditAPIException as err:
                logging.warning(f"Hit rate limit. Sleeping and trying again.\n{err}")
                time.sleep(args.wait_time)


