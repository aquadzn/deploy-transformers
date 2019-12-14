#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np

# Web app
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import uvicorn

# Removed XLM
from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, CTRLConfig
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer


def MODEL_CLASSES():
    '''Contains models and their tokenizer

    Returns
    -------
    dict
    '''
    return {
        'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
        'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
        'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
        'ctrl': (CTRLLMHeadModel, CTRLTokenizer)
    }

def ListModels():
    '''
    List available model names and model types
    '''
    print("Model type:\n\t" + " | ".join(MODEL_CLASSES().keys()))
    print("Model name:\n\t" + " | ".join(sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (OpenAIGPTConfig,GPT2Config, XLNetConfig, TransfoXLConfig, CTRLConfig)), ())))


class Model:
    '''The Model class contains functions used for text generation

    Parameters
    ----------
    model_type : str
        Define the type of model to use
    model_name : str
        Define the model to use  
    seed : int
        Define the seed to use [default: 42]  
    verbose : bool
        Enable or disable the logger [default: False]

    Attributes
    ----------
    verbose : bool
        This is where we store verbose
    model_type : str
        This is where we store model_type
    model_name : str
        This is where we store model_name
    model_class : str
        Model class to use
    model_tokenizer : str
        Model tokenizer to use
    device : str
        Device (CPU or GPU) to use
    n_gpu : int
        Number of GPU(s)
    set_seed : int
        This is where we store seed
    tokenizer
        Loaded tokenizer from chosen model
    model
        Loaded pretrained model from chosen model
    '''
    def __init__(self, model_type, model_name, seed=42, verbose=True):

        self.verbose = verbose
        if self.verbose == True:
            logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                datefmt = '%m/%d/%Y %H:%M:%S',
                                level = logging.INFO)
            self.logger = logging.getLogger(__name__)

        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model_class, self.tokenizer_class = MODEL_CLASSES()[self.model_type]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.set_seed(seed)

        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name)
        self.model = self.model_class.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def set_seed(self, seed):
        '''Seed the generator
        
        Parameters
        ----------
        seed : int
            Define a seed [default: 8080]
        
        Returns
        -------
        seed
        '''
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        '''Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        
        Parameters
        ----------
        seed : int
            Define a seed [default: 8080]
        logits :
            logits distribution shape (batch size x vocabulary size)
        top_k : int
            keep only top k tokens with highest probability (top-k filtering)
        top_p : float
            keep the top tokens with cumulative probability
        
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        
        Returns
        -------
        logits
        '''

        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(self, length, context, num_samples, temperature, top_k, top_p, repetition_penalty, is_xlnet): # Removed device='cpu' and XLM

        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in trange(length):

                inputs = {'input_ids': generated}
                if is_xlnet: 
                    # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                    # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                    input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=self.device)), dim=1)
                    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=self.device)
                    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=self.device)
                    target_mapping[0, 0, -1] = 1.0  # predict last token
                    inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

                outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

                # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for i in range(num_samples):
                    for _ in set(generated[i].tolist()):
                        next_token_logits[i, _] /= repetition_penalty
                    
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                if temperature == 0: # greedy sampling:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def PADDING_TEXT(self):
        '''Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
        in https://github.com/rusiaaman/XLNet-gen#methodology
        and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

        Returns
        -------
        str
        '''
        return """ In 1991, the remains of Russian Tsar Nicholas II and his family
        (except for Alexei and Maria) are discovered.
        The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
        remainder of the story. 1883 Western Siberia,
        a young Grigori Rasputin is asked by his father and a group of men to perform magic.
        Rasputin has a vision and denounces one of the men as a horse thief. Although his
        father initially slaps him for making such an accusation, Rasputin watches as the
        man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
        the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
        with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    def generate(
        self,
        length=20,
        prompt="",
        padding_text="",
        num_samples=1,
        temperature=1.0,
        top_k=0,
        top_p=0.9, 
        repetition_penalty=1.0,
        is_xlnet=False,
        stop_token=None):
        '''Generate predicted text

        Arguments
        ---------
        length : int
            Length of predicted text
        prompt : str
        padding_text : str
        num_samples : int
        temperature : float
            Temperature of 0 implies greedy sampling
        top_k : int
        top_p : float
        repetition_penalty : float
            Primarily useful for CTRL model; in that case, use 1.2
        is_xlnet : bool
            True if using XLNet, otherwise False
        stop_token : str
            Token at which text generation is stopped
        
        Returns
        -------
        text
            Predicted text based on prompt
        '''

        if length < 0 and self.model.config.max_position_embeddings > 0:
            args.length = self.model.config.max_position_embeddings
        elif 0 < self.model.config.max_position_embeddings < length:
            args.length = self.model.config.max_position_embeddings  # No generation bigger than model size 
        elif length < 0:
            length = int(10000)  # avoid infinite loop

        if self.verbose == True:
            if self.model_type in ["ctrl"]:
                if temperature > 0.7:
                    logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

        while True:

            raw_text = prompt if prompt else input("Model prompt >>> ")
            if self.model_type in ["transfo-xl", "xlnet"]:
                # Models with memory likes to have a long prompt for short inputs.
                raw_text = (padding_text if padding_text else self.PADDING_TEXT()) + raw_text
            context_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)

            if self.verbose == True:
                if self.model_type == "ctrl":
                    if not any(context_tokens[0] == x for x in self.tokenizer.control_codes.values()):
                        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")

            out = self.sample_sequence(
                length=length,
                context=context_tokens,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                is_xlnet=is_xlnet
            )

            out = out[:, len(context_tokens):].tolist()
            for o in out:
                text = self.tokenizer.decode(o, clean_up_tokenization_spaces=True)
                text = text[: text.find(stop_token) if stop_token else None]

                print(text)

            if prompt:
                break
        return text

class Website:
    '''The Website class contains the deploy function

    Parameters
    ----------
    model_type : str
        Define the type of model to use
    model_name : str
        Define the model to use    
    verbose : bool
        Enable or disable the logger [default: False]

    Attributes
    ----------
    model_type : str
        This is where we store model_type
    model_name : str
        This is where we store model_name
    verbose : bool
        This is where we store verbose
    '''
    def __init__(self, model_type, model_name, verbose=False):
        self.model_type = model_type
        self.model_name = model_name
        self.verbose = verbose

    def deploy(self, homepage_file='index.html', template_folder='templates', static_folder='static', host="0.0.0.0", port=8080):
        '''Deploy the model on a web-app at a given host:port
        
        Parameters
        ----------
        homepage_file : str 
            Homepage filename in the template directory [default: 'index.html']
        template_folder : str 
            Directory where are stored .html files [default: 'templates']
        static_folder : str
            Directory where are stored static files [default: 'static']
        host : str
            Bind app to this host [default: '0.0.0.0']
        port : int
            Bind app to this port [default: 8080]
        
        Returns
        -------
        app
            Running app
        
        '''
        templates = Jinja2Templates(directory=template_folder)

        app = Starlette(debug=False)
        app.mount('/static', StaticFiles(directory=static_folder), name='static')
        

        @app.route('/')
        async def homepage(request):
            '''Render homepage
            
            Arguments
            ---------
            requests

            Returns
            -------
            TemplateResponse()
                The homepage
            '''
            # template = "index2.html"
            # context = {"request": request}
            return templates.TemplateResponse(homepage_file, {"request": request})

        @app.route('/predict', methods=['GET', 'POST', 'HEAD'])
        async def predict(request):
            '''Render homepage
            
            Arguments
            ---------
            requests

            Returns
            -------
            JSONResponse()
                The predicted text
            '''
            if request.method == 'GET':
                params = request.query_params
            elif request.method == 'POST':
                params = await request.json()
            elif request.method == 'HEAD':
                return JSONResponse(
                    content={'text': ''},
                    headers={'Access-Control-Allow-Origin': '*'})
            
            text = model.generate(
                length=int(params.get('length', 20)),
                prompt=params.get('prompt', ''),
                temperature=float(params.get('temperature', 1.0))
            )

            return JSONResponse(
                content={'text': text},
                headers={'Access-Control-Allow-Origin': '*'})

        @app.exception_handler(404)
        async def not_found(request, exc):
            """
            Return an HTTP 404 page.
            """
            return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

        @app.exception_handler(404)
        async def server_error(request, exc):
            """
            Return an HTTP 500 page.
            """
            return templates.TemplateResponse("500.html", {"request": request}, status_code=500)


        model = Model(model_type=self.model_type, model_name=self.model_name, verbose=self.verbose)
        uvicorn.run(app, host=host, port=port)