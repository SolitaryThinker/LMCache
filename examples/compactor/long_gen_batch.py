import copy
import json
import os
import time

#from lmcache_vllm.vllm import LLM, SamplingParams
from lmcache_vllm.vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
context_file = os.path.join(os.pardir, 'ffmpeg.txt')
output_file = "offline_inference_outputs.jsonl"

context_text = None
with open(context_file, 'r') as f:
    context_text = f.read()
assert context_text is not None
tokenizer = AutoTokenizer.from_pretrained(model_name)

context_messages = [
    {
        "role": "user",
        "content": "You are a helpful assistant."
    },
    {
        "role": "assistant",
        "content": "Got it."
    },
]
user_inputs_batch = [
    # "What is FFmpeg?"
    # "Please include some details."
    # "Your answer should be around 5k words",
    """In the realm of Elennor, where the luminous moon Thalor watches over lands adorned with iridescent flora and sentient rivers, the fate of existence hangs by a delicate thread interwoven with the mystical essence known as Aether. This magical force courses through the land like invisible veins, granting life pulsating vibrancy and powers beyond the wildest imaginations. To harness such energy was a privilege bestowed upon the Aetherial Wizards, an ancient order dedicated to maintaining balance and valley peace across Elennor. Yet, this fragile peace was threatened.

The capital city, Elaris, was where these Aetherial Wizards convened. Elaris, with its spiraling towers of emerald and gold, nestled between the azure waters of Lake Sereth and the whispering Phaelon Forest, stood as a testament to Elennor’s architectural marvels. Magical creatures roamed freely through its enchanted streets. Tiny wisp-like sprites flittered like buzzing fireflies, and majestic gryphons perched on polished marble archways, observing the city below with regal indifference.

The people of Elaris lived in harmony, lavishing in the magic that intertwined with their daily routines. Floating marketplaces shimmered with wares from across Elennor, from delicately woven silks that changed hues to match the bearer’s mood, to fruits bursting with flavors of sweet stardust and crisp sunrise. Every corner echoed with laughter and song, a symphony of harmonious existence.

But even amidst such grandeur, shadows began to rise. Rumors of a hidden sect—The Nocturnals—reached the ears of the High Council of Wizards. The Nocturnals, a collective of renegade sorcerers disillusioned by the Council’s governing, sought to control the Aether for their own designs. They whispered of a prophecy buried in the depths of the past, one foretelling the arrival of an individual capable of commanding the Primordial Aether, a force even purer and more ancient than the gods themselves. This prophecy spoke of an eclipse that would shroud Thalor in red, a celestial event that would mark the rise of the Aether Wielder.

For decades, this prophecy was considered a myth, a bedtime story concocted to frighten young wizards. Yet now, the celestial signs became increasingly evident. Thalor’s glow dwindled each night, casting the lands in a shade of deepening dusk.

Amidst this growing turmoil, a young orphan named Arin, with no knowledge of his lineage, tended the gardens of the Emerald Sanctuary. Olena, his guardian and the Sanctuary’s keeper, raised him in the ways of herbalism and harmony with nature. Arin was adept at nurturing life, his touch invigorating even the most wilted of blossoms and the frailest of creatures. Unbeknownst to him, this ability was a vestige of something far greater.

His tranquil life shattered one fateful night when the earth trembled with an unfamiliar surge of energy. Woken by a brilliant, blinding light streaming through his window, Arin stumbled into the garden to find Olena conversing with an ethereal figure—an Aetherial Spirit. The specter, cloaked in swirling tendrils of light, declared Arin the prophesied Aether Wielder. The time had come for Arin to embrace his destiny.

Shocked and overwhelmed, Arin could scarcely fathom the enormity of this revelation. However, the urgency in the spirit’s voice granted him no moment for doubt. The all-important Eclipse was drawing near, and the Nocturnals were set on leveraging its power for ruinous ends.

Arin’s journey began at the crack of dawn, accompanied by Olena, who revealed herself as an Aetherial Guardian. Her purpose was to guide him to Elaris where he would train under the tutelage of the Aetherial Wizards. Their path was fraught with challenges, each designed to unlock the wells of power within Arin and prepare him for the trials ahead. They traversed the mystical terrains of Elennor—the thunderous peaks of the Karron Mountains, home to the enigmatic Storm Seekers; the enchanted Glimmering Sands desert, where visions of the future danced in the heat.

Arin learned quickly, harnessing the raw Aether within him to command the elements. He discovered a burgeoning ability to commune with the sentient rivers, understanding their wisdom locked in ancient flows, and to conjure flames born of primordial stars. Each step brought him closer to understanding his role in the unfolding fate of Elennor.

Meanwhile, Elaris buzzed with preparations to combat the rising threat. The council strategized tirelessly, calling upon allies from far and wide. The Luminal Knights, entrenched in their radiant armor, were assembled to guard the city against imminent attack. Citizens were trained in basic Aether manipulation, their hearts alight with courage and resolve.

As Arin’s prowess grew, so did the determination of the Nocturnals, leveraging dark magics twisted by their ambition. Their leader, a sorceress of formidable might and Arin’s nemesis, orchestrated plans to besiege Elaris and seize the Primordial Aether for chaos incarnate.

The climax builds towards the scheduled Eclipse, a time fraught with confrontation, betrayals, and unexpected alliances. Arin and his companions stand at the city’s heart, poised to battle forces that threaten not only Elaris but the eternal balance of Aether itself.

In an epic crescendo, the question remains: can Arin, a simple gardener of mysterious heritage, rise to the fore and become the champion Elennor so desperately needs?
"""
]
user_inputs_batch = [
    "What is FFmpeg?"
    "Please include some details."
    "Your answer should be around 5k words",
]


def get_context_length(tokenizer, context_messages):
    return len(tokenizer.apply_chat_template(context_messages, tokenize=False))

def get_prompt_length(tokenizer, prompt):
    return len(tokenizer.encode(prompt))

def gen_prompts(tokenizer, context_messages, user_inputs_of_batch):
    generated_prompts = []
    for user_input in user_inputs_of_batch:
        copyed_context_messages = copy.deepcopy(context_messages)
        copyed_context_messages.append({"role": "user", "content": user_input})
        generated_prompts.append(
            tokenizer.apply_chat_template(copyed_context_messages,
                                          tokenize=False))
    return generated_prompts


def append_outputs(output_file_name, outputs, context_length, time_taken):
    user_inputs = []
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        user_input = prompt[context_length:]
        user_inputs.append(user_input)
        generated_text = output.outputs[0].text
        generated_texts.append(f"{generated_text!r}")
    json_dict = {
        "user_inputs": user_inputs,
        "generated_texts": generated_texts,
        "time in seconds": time_taken
    }
    with open(output_file_name, "a") as f:
        f.write(json.dumps(json_dict) + '\n')


context_length = get_context_length(tokenizer, context_messages)
# Create a sampling params object.

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=600)

prompts = gen_prompts(tokenizer, context_messages, user_inputs_batch)
l = get_prompt_length(tokenizer, prompts[0])
# Create an LLM.
llm = LLM(model=model_name,
          gpu_memory_utilization=0.8,
          enable_chunked_prefill=True,
          max_model_len=32768,
          max_num_batched_tokens=5,
          max_num_seqs=5,
          enforce_eager=True)

# Clear output file.
with open(output_file, "w") as f:
    pass

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

print(f"prompts: {prompts}")
# print('context_length', context_length)
print('prompt length', l)
# print(f"prompts: {prompts}")
t1 = time.perf_counter()
second_outputs = llm.generate(prompts, sampling_params)
t2 = time.perf_counter()
print(f"\n\nRequest Time: {t2 - t1} seconds\n\n")
print(f"Output: {second_outputs}")