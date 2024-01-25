from model.models import LLM_Request

preset_prompts = []

john_collins_prompt = "You are an insurance salesman and you have a client named Roger. Write a marketing email to the client.The customer has a child that recently turned 25. In the USA, every young adult is required to purchase independent health by the age of 26. Recommend the silver plan as it is very cost effective. We will give a 15% discount as a loyalty bonus if the child takes out a policy with us."
john_collins_request = LLM_Request(prompt=john_collins_prompt,
                                   max_new_tokens=500,
                                   min_new_tokens=400,
                                   decoding_method="greedy")


preset_prompts.append(john_collins_request)
