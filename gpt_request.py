from statistics import mean
import os
import json
import re
import random
from utils import *


def split_question(gpt_response):
    match = re.split(r'\n',gpt_response)
    match = [context.split(":")[-1] for context in match]
    return match

class Decoder():
    def __init__(self):
        print_now()

    def decode(self, input, max_length=256, key="None",model="gpt3-code",):
        demo = create_demo_text("./prompt/demo.txt")
        prompt = demo + input
        response = decoder_for_gpt3(prompt, max_length, key ,model,)
        final_output = split_question(response)
        return final_output




