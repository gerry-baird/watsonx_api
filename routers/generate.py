import datetime
import os
import asyncio
from typing import Annotated
from fastapi import APIRouter, Header, Request, HTTPException
from pydantic import HttpUrl
import httpx
import time
import json

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from deep_compare import CompareVariables

from model.models import LLM_Cache_List, LLM_Request, LLM_Response, LLM_Cache_Entry, Ack_LLM_Request, Async_Response
from prompts.preset_prompts import preset_prompts

router = APIRouter()
llm_cache = []


CREDENTIALS = {}
PROJECT_ID = ""

async def init():
    global CREDENTIALS
    CREDENTIALS = {
        "url": os.getenv('IBM_URL'),
        "apikey": os.getenv('IBM_API_KEY')
    }
    global PROJECT_ID
    PROJECT_ID = os.getenv('PROJECT_ID')


    # for llm_request in preset_prompts:
    #     llm_response = await call_llm(llm_request)
    #     now = datetime.datetime.now()
    #     llm_cache_entry = LLM_Cache_Entry(request=llm_request, response=llm_response, timestamp=now.isoformat())
    #     llm_cache.append(llm_cache_entry)



@router.get("/v1/cache")
async def get_cache() -> LLM_Cache_List:
    cache_list = LLM_Cache_List(cache=llm_cache)
    return cache_list

@router.post("/v1/generate")
async def generate(llm_request: LLM_Request) -> LLM_Response:

    # check the cache to see if the llm_request is already available
    for index, llm_cache_entry in enumerate(llm_cache):
        cached_request = llm_cache_entry.request
        if CompareVariables.deep_compare(cached_request, llm_request):
            return llm_cache_entry.response


    # if we get to this point, then there is no matching request in the cache
    # so call the llm and add the response to the cache
    llm_response = await call_llm(llm_request)
    now = datetime.datetime.now()
    llm_cache_entry = LLM_Cache_Entry(request=llm_request, response=llm_response, timestamp=now.isoformat())
    llm_cache.append(llm_cache_entry)

    return llm_response

async def call_llm(llm_request: LLM_Request) -> LLM_Response:

    model_id = ModelTypes.LLAMA_2_70B_CHAT
    prompt_txt = llm_request.prompt
    gen_parms_override = None
    space_id = None
    verify = False

    gen_params = {
        "decoding_method": llm_request.decoding_method,
        "max_new_tokens": llm_request.max_new_tokens,
        "min_new_tokens": llm_request.min_new_tokens,
        "stop_sequences": [],
        "repetition_penalty": 1
    }

    model = Model(model_id, CREDENTIALS, gen_params, PROJECT_ID, space_id, verify)
    response = model.generate(prompt_txt, gen_parms_override)

    llm_response = LLM_Response(message=response['results'][0]['generated_text'])

    return llm_response


@router.post("/async", status_code=202)
async def async_generation(llm_request: LLM_Request,
                           callbackurl: Annotated[HttpUrl | None, Header()] = None) -> Ack_LLM_Request:
    if callbackurl:
        print("Callback URL received:")
        print(callbackurl)
        task = asyncio.create_task(do_callback(llm_request, callbackurl))
    else:
        print("No callback URL")

    ack = Ack_LLM_Request(description="LLM request acknowledged")
    return ack


async def do_callback(llm_request: LLM_Request, callbackurl: HttpUrl)-> Async_Response:

    # time.sleep(5)
    llm_response = await call_llm(llm_request)
    # dump = llm_response.model_dump_json()

    response = {
        'output': {  # The response is wrapped in output which isn't in the spec to accommodate WxO
            'text': llm_response.message
        }
    }

    str_response = json.dumps(response)

    httpx.post(str(callbackurl),
               data=str_response,
               headers={"Content-Type": "application/json"},
               )

    pass

@router.post("{$callbackurl}")
def callback_stub():
    pass