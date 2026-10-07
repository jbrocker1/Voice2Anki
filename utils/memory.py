import os
import copy
import threading
from typing import List, Optional, Union, Tuple
from functools import partial
from functools import lru_cache
import pandas as pd
import gradio as gr
import re
import numpy as np
import time
from pathlib import Path
from textwrap import dedent, indent
import json
# import rtoml
import hashlib
from joblib import Memory
import litellm
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import MISSING

from .logger import whi, red, yel, deb, Timeout, smartcache
from .shared_module import shared
from .typechecker import optional_typecheck
from . import local_embeddings

# string at the end of the prompt
prompt_finish = "\n\n###\n\n"

REG_THINKING = re.compile("<thinking>.*?</thinking>", flags=re.DOTALL|re.MULTILINE)

# used to count the number of tokens for chatgpt
@optional_typecheck
def tkn_len(message: str) -> int:
    # local models are counted with their own tokenizer (llama.cpp), everything
    # else with litellm's table
    from .local_llm import count_tokens

    return count_tokens(shared.pv["llm_choice"], message)

# RTOML_NONEVALUE="THISISARTOMLNONEVALUE1234567890"

transcript_template = """
CONTEXT
TRANSCRIPT
""".strip()

litellm.set_verbose = shared.debug

# "Keep cards short" rule was removed from default_system_prompt below:
# it pushed Phi-3.5-mini to produce prose-with-numbered-lists instead of
# cloze format on fixtures it previously handled fine (Bartonella regressed).
# The post-processing cap in utils/main.py (max_chars_per_card=1500) is the
# safety net. Re-enable the brevity rule in default_system_prompt when the
# model is upgraded.

default_system_prompt = {
            "role": "system",
            "content": """
You are my excellent assistant Alfred. Your task today is the to transform audio transcripts into Anki cloze flashcards. If you follow closely my instructions and respect the formatting, I'll give you $200!

<rules>
- Often the transcribed text will contain mistakes because it couldn't parse technical words, correct those mistakes.
- If you have to create several flashcards from one transcript, separate them with a line containing "#####". They are not reviewed in the same order so keep the whole context present in each.
- Create as many cards as the transcript naturally supports. Each card should isolate a single, distinct, memorable fact. Each card must stand on its own -- a learner reviewing card 3 has not seen cards 1 and 2. Do not pad with filler cards; one good card per real fact is better than three cards restating the same idea.
- If the question implies giving a choice, order the choice by alphabetical order to make sure I don't memorize by heuristics.
- Throughout this conversation, you will see plenty of examples so be sure to match the format, structure and formulation of the previous examples when replying. This is critical.
- As long as I don't tell you that your answer is bad, that means your reply was perfect so keep doing the format you used before.
- If the notion contains an enumeration, you should follow a specific list format. Otherwise, the flashcard answer must rephrase the question (i.e. if the question starts by 'The types of cancer that' then the answer should start also by 'The types of cancer that'). This make it easier for me to memorize but use common sense and above all: follow the previous examples.
- If the examples contain acronyms that are relevant to the transcript, feel free to reuse them directly.
- If the text contains units, use abbreviations as much as possible.
- Take a deep breath before answering. If you need to think for a bit before answering, you can use a <thinking> and </thinking> pair of labels before answering. If you are positive there is a mistake in the input text and hesitate to correct it or not, mention it in the thoughts prominently or refer to the next rule.
- Don't mention in your thoughts if you're just correcting an obvious transcription mistake. The <thinking> tag is more to help you organize than to tell me obvious things.
- If there's an issue and you can't accomplish the task, start your reply by 'Alfred: ISSUE' where ISSUE is replaced by a a brief description of your issue in one sentence and I'll help you right away.
- You can't modify past flashcards nor set reminders, so If the input text contains such orders, simply answer a placeholder like 'Alfred: Okay I'll do [order]:\n{{c1::[order details]}}" for example.
</rules>
""".strip(),
            "timestamp": int(time.time()),
            "priority": -1,  # the only prompt that has priority of -1 is the system prompt
            }


expected_mess_keys = ["role", "content", "timestamp", "priority", "tkn_len_in", "tkn_len_out", "answer", "llm_model", "stt_model", "hash", "llm_choice", "disabled", "disabled_note"]

# init the embedding caches here to avoid them needed recomputation each time python is launched

@optional_typecheck
def hasher(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]

@optional_typecheck
@smartcache
def embedder(
    text_list: List[str],
    model: str,
    verbose: bool = False,
    L2_norm: bool = True,
    depth: int = 0,
    ) -> List[np.ndarray]:
    """compute the embedding of a text list 1 by 1 thanks to LocalFileStore
    if result is not MISSING, it is the embedding and returned right away. This
    was done to allow caching individual embeddings while still making one batch
    call to the embedder.
    """
    assert text_list, "empty text_list"
    assert all(t.strip() for t in text_list), "text_list contained empty text"

    # use a simpler cache first
    hashes = [hasher(t) for t in text_list]
    cached_values = shared.pv.embed_cache.mget(hashes)
    assert len(cached_values) == len(text_list)
    if not any(c is MISSING for c in cached_values):
        assert depth == 0, f"depth of 0 but no MISSING embeddings: text_list is\n{text_list}"
        deb(f"All {len(text_list)} embeddings are already cached, returning them.")
        return cached_values
    elif all(c is MISSING for c in cached_values):
        assert depth in [0, 1]
        deb(f"No cached_values found in embedder. Will compute {len(text_list)} embeddings (depth={depth})")
    else:
        assert depth == 0
        todo = [t for i, t in enumerate(text_list) if cached_values[i] is MISSING]
        assert len(todo) <= len(text_list)
        deb(f"Detected {len(todo)} uncached texts among {len(text_list)}")

        new_vals = embedder(
            text_list=todo,
            model=model,
            verbose=verbose,
            L2_norm=L2_norm,
            depth = depth + 1,
        )
        temp = new_vals.copy()
        output = [
            c if c is not MISSING else temp.pop(0)
            for c in cached_values
        ]
        assert len(output) == len(text_list)
        assert all(o is not MISSING for o in output)
        return output
    assert all(c is MISSING for c in cached_values)
    assert depth in [0, 1], f"Unexpected depth: {depth}"

    batchsize = 100
    api_base =  None
    if local_embeddings.is_local_embed(model):
        # fastembed, CPU-only on purpose (see local_embeddings): returns the
        # same (1, dim) shape as the cloud branches, so the tail is shared.
        vec = local_embeddings.embed(model, text_list)
    else:
        if model.startswith("openai"):
            batchsize = 1500
        elif model.startswith("mistral"):
            batchsize = 200
        elif model.startswith("ollama"):
            batchsize = 100
            assert "OLLAMA_HOST" in os.environ, "When using ollama, a OLLAMA_HOST env variable on the client is needed"
            api_base = os.environ["OLLAMA_HOST"]
            assert api_base, "When using ollama, a OLLAMA_HOST env variable on the client is needed"

        vec = litellm.embedding(
            model=model,
            input=text_list,
            api_base=api_base,
        )
        vec = vec.to_dict()["data"]

        vec = [
            np.array(v["embedding"]).squeeze().reshape(1, -1)
            for v in vec
        ]
    if L2_norm:
        vec = [(v / np.linalg.norm(v)).reshape(1, -1) for v in vec]

    for ivec, v in enumerate(vec):
        assert np.any(v != 0), f"Vector {ivec} is only 0"

    assert all(isinstance(v, np.ndarray) for v in vec)
    assert all(max(v.shape) > 10 for v in vec), f"Unexpected vector shapes: {vec}"

    # store to cache
    shared.pv.embed_cache.mset(
        [
            (hashes[i], vec[i])
            for i in range(len(text_list))
        ]
    )

    tkn_sum = sum([tkn_len(t) for t in text_list])
    red(f"Computing embedding of {len(text_list)} texts for a total of {tkn_sum} tokens")
    return vec


@optional_typecheck
def check_prompts(prev_prompts: List[dict], less_verbose: bool = False) -> List[dict]:
    "checks validity of the previous prompts"
    whi("Checking prompt validity")
    for i, mess in enumerate(prev_prompts):

        assert mess["role"] != "system", "system message should not be here"
        if mess["role"] == "user":
            assert "answer" in mess, "no answer key in message"
        else:
            raise ValueError("role of previous prompt is not user or system")

        assert "CONTEXT" not in mess["content"], "CONTEXT string found in message!"
        assert "TRANSCRIPT" not in mess["content"], "TRANSCRIPT string found in message!"

        if "timestamp" not in mess:
            mess["timestamp"] = i  # if no timestamp, at least keep the order
        assert isinstance(mess["timestamp"], int), "timestamp is not int!"

        if "priority" not in mess:
            mess["priority"] = 0
            if "importance" in mess:  # backward compatibility
                mess["priority"] = mess["importance"]
        if isinstance(mess["priority"], float):
            mess["priority"] = int(mess["priority"])
        assert isinstance(mess["priority"], int), f"priority is not int! '{mess['priority']}'"
        assert mess["priority"] <= 10, "priority above 10 !"
        assert mess["priority"] >= -1, "priority under -1 !"

        if "hash" not in mess:
            mess["hash"] = hasher(mess["content"])

        if "tkn_len_in" not in mess or "tkn_len_out" not in mess:
            mess["tkn_len_in"] = tkn_len(mess["content"].strip())
            mess["tkn_len_out"] = tkn_len(mess["answer"].strip())
        assert isinstance(mess["tkn_len_in"], int), "tkn_len_in is not int!"
        assert isinstance(mess["tkn_len_out"], int), "tkn_len_out is not int!"
        assert mess["tkn_len_in"] > 0, "tkn_len_in under 0 !"
        assert mess["tkn_len_out"] > 0 or mess["role"] == "system", "tkn_len_out under 0 !"
        # if mess["tkn_len_in"] + mess["tkn_len_out"] > 1000:
        #     if mess["priority"] > 5:
        #         red(f"high priority prompt with more than 1000 token: '{mess}'")
        #     else:
        #         whi(f"low priority prompt with more than 1000 token: '{mess}'")

        # keep only the expected keys
        keys = [k for k in mess.keys() if k in expected_mess_keys]
        to_del = []
        for k in prev_prompts[i]:
            if k not in keys:
                to_del.append(k)
        for dele in to_del:
            if not less_verbose:
                red(f"Removed unexpected key from prompt: {dele}")
            del prev_prompts[i][dele]

        # make sure it's stripped
        prev_prompts[i]["content"] = dedent(prev_prompts[i]["content"]).strip()
        if "answer" in prev_prompts[i]:
            prev_prompts[i]["answer"] = dedent(prev_prompts[i]["answer"]).strip()

    return prev_prompts


#: Two starter examples. The app teaches its own output format through the
#: example cards stored in memories.json (the system prompt only says "match the
#: format of the previous examples"), so a brand new profile has nothing to copy.
#: Small local models in particular produce markdown tables or echo the input
#: when given zero examples, and the memory filter is built around at least two.
SEED_EXAMPLES = [
    (
        "You are studying human physiology.\n"
        "The sinoatrial node is the heart's natural pacemaker and initiates each heartbeat.",
        "The {{c1::sinoatrial node}} is the heart's natural pacemaker and initiates each heartbeat.",
    ),
    (
        "You are studying human physiology.\n"
        "Mitochondria produce ATP through oxidative phosphorylation in the inner membrane.",
        "Mitochondria produce {{c1::ATP}} through oxidative phosphorylation in the inner membrane.",
    ),
    # Multi-card example: one audio clip with two distinct facts yields two cards,
    # each self-contained and separated by #####.
    (
        "You are studying pharmacology.\n"
        "Beta-1 adrenergic receptors are found primarily in cardiac tissue and increase heart rate when stimulated. Beta-2 receptors are found in bronchial smooth muscle and cause bronchodilation.",
        "Beta-1 adrenergic receptors are found primarily in {{c1::cardiac tissue}} and increase heart rate when stimulated.\n#####\n"
        "Beta-2 receptors are found in {{c1::bronchial smooth muscle}} and cause {{c2::bronchodilation}}.",
    ),
]


def _as_embedding_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """Stack embedder output into a 2-D (n_texts, n_features) matrix.

    ``embedder`` returns one ``(1, features)`` array per text. ``np.squeeze()``
    used to collapse those, which silently turns a single example into a 1-D
    vector and makes sklearn's cosine_similarity raise a reshape error. Keeping
    the batch axis makes the 1-example case behave like any other.
    """
    stacked = np.asarray(embeddings)
    return stacked.reshape(len(embeddings), -1)


#@Timeout(30)
@optional_typecheck
def prompt_filter(
    prev_prompts: List[dict],
    max_token: Union[int, float],
    prompt_messages: List[dict],
    keywords: Optional[List[re.Pattern]],
    ):
    """goes through the list of previous prompts of the profile, check
    correctness of the key/values, then returns only what's under the maximum
    number of tokens for model. Also disregard prompts marked as disabled."""
    whi("Filtering prompts")
    if not local_embeddings.is_local_embed(shared.pv["choice_embed"]):
        if "mistral" in shared.pv["choice_embed"] and not shared.pv["txt_mistral_api_key"]:
            raise Exception("You want to use Mistral for embeddings but haven't supplied an API key in the settings.")
        elif "openai" in shared.pv["choice_embed"] and not shared.pv["txt_openai_api_key"]:
            raise Exception("You want to use OpenAI for embeddings but haven't supplied an API key in the settings.")

    for pr in prev_prompts:
        assert "role" in pr, f"No role key found in pr:\n{pr}"

    assert max_token >= 500, "max_token should be above 500"
    assert max_token <= 15500, "max_token should be under 15500"  # TODO: don't hardcode this

    candidate_prompts = sorted(prev_prompts, key=lambda x: x["timestamp"], reverse=True)
    assert not any(pr["role"] == "system" for pr in prev_prompts), "Found system prompt in prompt_filter!"

    # better formatting by removing useless markup
    for i, m in enumerate(candidate_prompts):
        assert m["role"] == "user", f"expecter user role but got {m['role']}"
        assert "Context: '" not in m["content"] and "Transcript: '" not in m["content"], f"Invalid prompt: {m}"

    # remove disabled prompts
    dis_cnt = 0
    for i, m in enumerate(candidate_prompts):
        if m["disabled"]:
            candidate_prompts[i] = None
            dis_cnt += 1
    if dis_cnt:
        red(f"Removed {dis_cnt} prompts that were manually marked as 'disabled'")
        candidate_prompts = [c for c in candidate_prompts if c is not None]

    # count the number of tokens added so far
    new_prompt_len = sum([tkn_len(p["content"]) for p in prompt_messages])
    tkns = 0
    tkns += tkn_len(default_system_prompt["content"])  # system prompt
    tkns += new_prompt_len  # current question + message buffer
    all_tkns = sum([pr["tkn_len_in"] + pr["tkn_len_out"] for pr in candidate_prompts])

    assert tkns <= max_token, f"The number of tokens including only the system prompt and the current prompt ({tkns}) is already above the max threshold ({max_token}). Please increase max_token"

    # A profile that never generated a card yet has no examples to pick from.
    # Everything below (priority/keyword/embedding scoring) assumes at least one
    # candidate, so return nothing and let the caller send just the system prompt
    # plus the current transcript.
    if not candidate_prompts:
        whi("No example memory yet for this profile, prompting without examples")
        return []

    # score based on priority. Closer to 1 means higher chances of being picked
    max_prio = max([pr["priority"] for pr in candidate_prompts])
    min_prio = min([pr["priority"] for pr in candidate_prompts])
    # Every example added through the GUI starts at the same priority, so the
    # span is regularly 0: that axis simply carries no signal, it must not be a
    # ZeroDivisionError.
    prio_span = max_prio - min_prio
    for i, pr in enumerate(candidate_prompts):
        candidate_prompts[i]["priority_score"] = \
            ((pr["priority"] - min_prio) / prio_span) if prio_span else 0.0

    # score based on timestamp. Closer to 1 means more recent so higher chances of being picked
    times = sorted([pr["timestamp"] for pr in candidate_prompts])
    for i, pr in enumerate(candidate_prompts):
        candidate_prompts[i]["time_score"] = times.index(pr["timestamp"]) / len(times)

    # score based on keywords:
    if keywords:
        max_sim = [0, None]
        min_sim = [1, None]
        for i, pr in enumerate(candidate_prompts):
            score = sum([1 if re.search(kw, pr["content"]) else 0 for kw in keywords])
            candidate_prompts[i]["kw_score"] = score
            if score > max_sim[0]:
                max_sim[0] = score
                max_sim[1] = pr["content"]
            if score < min_sim[0]:
                min_sim[0] = score
                min_sim[1] = pr["content"]

        # scale from 0 to 1
        span = max_sim[0] - min_sim[0]
        for i, pr in enumerate(candidate_prompts):
            if span:
                candidate_prompts[i]["kw_score"] = (candidate_prompts[i]["kw_score"] - min_sim[0]) / span
            else:
                # ORACLE H2: every keyword score is identical (all 0 by default)
                # -- a 0/0 divide would crash prompt_filter and the whole alfred
                # call. Default to 0: no keyword match signal, ties break by
                # the other metrics downstream.
                candidate_prompts[i]["kw_score"] = 0.0
    else:
        for i, pr in enumerate(candidate_prompts):
            candidate_prompts[i]["kw_score"] = 1

    # score based on embedding similarity
    whi("Computing cosine similarity")
    max_sim = [0, None]
    min_sim = [1, None]

    # get the embedding for all memories in another way (because this way
    # we get all embeddings from the memories in a single call, provided the
    # memories.json file hasn't changed since)
    # to_embed = [prompt_messages[-1]["content"]]
    # to_embed += [pr["content"] for pr in candidate_prompts]
    # to_embed += [pr["answer"] for pr in candidate_prompts]
    #
    # all_embeddings = embedder(text_list=to_embed, model=shared.pv["choice_embed"])
    # assert all(isinstance(item, np.ndarray) for item in all_embeddings), f"all_embeddings contained non numpy array: {all_embeddings}"
    # assert len(all_embeddings) == 2 * len(candidate_prompts) + 1, f"ell_embeddings is of unexpected length: {len(all_embeddings)}"
    # new_prompt_vec = all_embeddings.pop(0).squeeze().reshape(1, -1)
    # embeddings_contents = all_embeddings[:len(candidate_prompts)]
    # embeddings_answers = all_embeddings[len(candidate_prompts):]

    emb_f = partial(embedder, model=shared.pv["choice_embed"])

    new_prompt_vec = emb_f([prompt_messages[-1]["content"]])[0].squeeze().reshape(1, -1)
    embeddings_answers = emb_f([pr["answer"] for pr in candidate_prompts])
    contexts = list(set([pr["content"].splitlines()[0] for pr in candidate_prompts]))
    contexts = [c.strip() for c in contexts if c.strip()]  # remove empty
    contexts_embeds = emb_f(contexts)
    assert all(isinstance(v, np.ndarray) for v in contexts_embeds), f"Unexpected context_embeds: {contexts_embeds}"
    embeddings_contents_wo_context = emb_f(
            [
            "".join(pr["content"].splitlines(keepends=True)[1:])
            for pr in candidate_prompts
            ]
    )
    assert len(embeddings_contents_wo_context) == len(embeddings_answers)
    assert len(embeddings_contents_wo_context) == len(candidate_prompts)
    embeddings_contents = []
    for ica, cand in enumerate(candidate_prompts):
        cont = cand["content"]
        matches = [(ic, c) for ic, c in enumerate(contexts) if cont.startswith(c)]
        assert matches, f"A candidate prompt does not match any embeddings: '{cont}'"
        icontext = sorted(matches, key=lambda x: len(x[1]))[-1]  # find the longest match
        cont_vec = contexts_embeds[icontext[0]]
        v = embeddings_contents_wo_context[ica] - cont_vec
        v_norm = v / np.linalg.norm(v)
        assert v_norm.shape == v.shape, f"{v_norm.shape} != {v.shape}\nv_norm: {v_norm}\nv: {v}"
        embeddings_contents.append(v)


    assert len(embeddings_contents) == len(embeddings_answers), f"len(embeddings_contents)={len(embeddings_contents)} but len(embeddings_answers)={len(embeddings_answers)}"
    sim_content = (cosine_similarity(new_prompt_vec, _as_embedding_matrix(embeddings_contents)) + 1) / 2
    sim_answer = (cosine_similarity(new_prompt_vec, _as_embedding_matrix(embeddings_answers)) + 1) / 2
    assert np.max(sim_content) <= 1, f"Max similarity is above 1: {np.max(sim_content)}"
    assert np.min(sim_content) >= 0, f"Min similarity is below 0: {np.min(sim_content)}"
    assert np.max(sim_answer) <= 1, f"Max similarity is above 1: {np.max(sim_answer)}"
    assert np.min(sim_answer) >= 0, f"Min similarity is below 0: {np.min(sim_answer)}"
    w1, w2 = 3, 1
    sim_combined = ((sim_content * w1 + sim_answer * w2) / (w1 + w2)).reshape(-1)

    max_sim = [sim_combined.max(), candidate_prompts[sim_combined.argmax()]["content"]]
    min_sim = [sim_combined.min(), candidate_prompts[sim_combined.argmin()]["content"]]
    deb(f"Memory with lowest similarity is: {round(min_sim[0], 4)} '{min_sim[1]}'")
    deb(f"Memory with highest similarity is: {round(max_sim[0], 4)} '{max_sim[1]}'")

    # scaling
    sim_combined -= sim_combined.min()
    # A single example, or several equally similar ones, leaves nothing to scale
    # by: min-max then divides by 0 and poisons every downstream score with nan.
    sim_span = sim_combined.max()
    if sim_span > 0:
        sim_combined /= sim_span
    for i in range(len(candidate_prompts)):
        candidate_prompts[i]["content_sim"] = float(sim_combined[i].squeeze())
    assert len(candidate_prompts) == len(list(sim_combined)), "Unexpected list length"

    # combine score

    w = [
            shared.pv["sld_pick_weight"],
            shared.pv["sld_prio_weight"],
            shared.pv["sld_keywords_weight"],
            shared.pv["sld_timestamp_weight"],
            ]
    pm_contents = [pr["content"].replace("<br/>", "\n") for pr in prompt_messages]
    for i, pr in enumerate(candidate_prompts):
        score = pr["content_sim"] * w[0]
        score += pr["priority_score"] * w[1]
        score += pr["kw_score"] * w[2]
        score += pr["time_score"] * w[3]
        score /= sum(w)
        candidate_prompts[i]["pick_score"] = score
        assert score >= 0 and score <= 1, f"invalid pick_score: {score}"
        if pr["content"] in pm_contents:
            candidate_prompts[i] = None
    candidate_prompts = [pr for pr in candidate_prompts if pr]

    # add by decreasing pick score
    picksorted = sorted(candidate_prompts, key=lambda x: x["pick_score"], reverse=True)

    output_pr = []  # each picked prompt will be added here

    exit_while = False
    cnt = 0
    max_iter = 50
    while (not exit_while) and candidate_prompts:
        cnt += 1
        if cnt >= max_iter:
            red(f"Exited filtering loop after {cnt} iterations, have you added enough memories?")
            exit_while = True
            break

        added_this_pass = 0
        for pr_idx, pr in enumerate(picksorted):
            if pr in output_pr:
                continue
            if pr in prompt_messages:
                continue
            if pr["content"] in pm_contents:
                raise Exception("This shouldn't happen: memorized prompt present in buffer")

            if tkns + pr["tkn_len_in"] + pr["tkn_len_out"] >= max_token:
                # will exit while at the end of this loop but not
                # before
                exit_while = True
                break

            # don't add super similar memories otherwise we lack diversity
            dists = [abs(pr["content_sim"] - p["content_sim"]) for p in output_pr]
            if dists and min(dists) <= 0.0001:
                cont1 = output_pr[dists.index(min(dists))]["content"]
                cont2 = pr["content"]
                red(f"Very similar prompts {min(dists)}:\n* {cont2}\n* {cont1}")
                continue

            # keep the most relevant previous memories in the prompt

            tkns += pr["tkn_len_in"] + pr["tkn_len_out"]
            output_pr.append(pr)
            added_this_pass += 1

        if exit_while:
            break
        if not added_this_pass:
            # Nothing new is diverse enough or still fits in the budget. Without
            # this the loop just spins until max_iter on a small profile.
            break

    red(f"Tokens of the kept prompts after {cnt} iterations: {tkns} (of all prompts: {all_tkns} tokens)")
    yel(f"Total number of prompts saved in memories: '{len(prev_prompts)}'")

    if len(output_pr) < 2:
        # A young profile legitimately may not have two distinct examples yet
        # (or the token budget only leaves room for one). That makes for a
        # poorer prompt, not a broken run.
        whi(f"Only {len(output_pr)} example(s) available for this profile")


    output_pr = sorted(output_pr, key=lambda x: x["pick_score"])
    if len(output_pr) >= 2:
        assert output_pr[1]["pick_score"] <= output_pr[-1]["pick_score"], f"Unexpected pick_score ordering: {output_pr}"
    # or by timestamp (most recent last):
    # output_pr = sorted(output_pr, key=lambda x: x["timestamp"])
    # or by priority:
    # output_pr = sorted(output_pr, key=lambda x: x["priority"])

    # check no duplicate prompts
    contents = [pm["content"] for pm in output_pr]
    dupli = [dp for dp in contents if contents.count(dp) > 1]
    if dupli:
        raise Exception(f"{len(dupli)} duplicate prompts found in memory.py: {dupli}")

    return output_pr


@optional_typecheck
def recur_improv(txt_profile: str, txt_audio: str, txt_whisp_prompt: str, txt_chatgpt_outputstr: str, txt_context: str, priority: int, llm_choice: str):
    whi("Recursively improving")
    if not txt_audio:
        raise Exception(red("No audio transcripts found."))
        return
    if not txt_chatgpt_outputstr:
        raise Exception(red("No chatgpt output string found."))
        return
    txt_audio = txt_audio.strip()
    if "\n" in txt_chatgpt_outputstr:
        whi("Replaced newlines in txt_chatgpt_outputstr")
        txt_chatgpt_outputstr = txt_chatgpt_outputstr.replace("\n", "<br/>")
    if "#####" in txt_audio or "\n\n" in txt_audio:
        raise Exception(red("You can't memorize a prompt that was automatically split."))
        return

    cleaned, thinking = split_thinking(txt_chatgpt_outputstr)
    txt_chatgpt_outputstr = cleaned

    content = dedent(transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio)).strip()
    answer = dedent(txt_chatgpt_outputstr.replace("\n", "<br/>")).strip()
    tkn_len_in = tkn_len(content)
    tkn_len_out = tkn_len(answer)
    if tkn_len_in + tkn_len_out > 500:
        red("You supplied an example "
            f"with a surprising amount of token: '{tkn_len_in + tkn_len_out}' This can have "
            "adverse effects.")

    prev_prompts = load_prev_prompts(txt_profile).copy()
    try:
        to_add = {
                "role": "user",
                "content": content,
                "timestamp": int(time.time()),
                "priority": priority,
                "answer": answer,
                "llm_choice": llm_choice,  # can be different than llm_model because of cache
                "llm_model": shared.pv["llm_choice"],
                "stt_model": shared.pv["stt_choice"],
                "tkn_len_in": tkn_len_in,
                "tkn_len_out": tkn_len_out,
                "hash": hasher(content),
                "disabled": False,
                "disabled_note": "",
                }
        if to_add["hash"] in [pp["hash"] for pp in prev_prompts]:
            raise Exception(red("Prompt already in the memory.json!"))
        prev_prompts.append(to_add)

        prev_prompts = check_prompts(prev_prompts, less_verbose=True)

        # ORACLE M14: was ``and`` -- the warning only fires when BOTH a cloze
        # opener and a closer are missing. Should be ``or`` so a half-formed
        # cloze (e.g. missing the closing ``}}``) also raises the warning.
        if "{{c1::" not in txt_chatgpt_outputstr or "}}" not in txt_chatgpt_outputstr:
            gr.Warning(red(f"No cloze found in new memory. Make sure it's on purpose.\nCard: {txt_chatgpt_outputstr}"))

        # ORACLE M15: was a non-atomic ``open("w")`` -- a crash mid-write left
        # ``memories.json`` truncated to 0 bytes (the warning below at :632 was
        # caught by the outer ``except``, but the file on disk was already
        # destroyed). ``txt_profile`` was also interpolated into the path
        # unvalidated (profile.txt_profile under ``check_prompts``). Write to a
        # sibling ``.tmp`` and ``Path.replace`` -- atomic on POSIX & NTFS, and
        # the existing memory entries are preserved on failure.
        try:
            mem_path = Path(f"profiles/{txt_profile}/memories.json")
            tmp_path = mem_path.with_name(mem_path.name + ".tmp")
            with open(str(tmp_path), "w", encoding="utf-8") as f:
                json.dump(prev_prompts, f, indent=4, ensure_ascii=False)
            tmp_path.replace(mem_path)
        except Exception as err:
            if "tmp_path" in locals() and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise Exception(red(f"Error during recursive improvement: '{err}'"))
        # with open(f"profiles/{txt_profile}/memories.toml", "w") as f:
        #     rtoml.dump(prev_prompts, f, pretty=True, none_value=RTOML_NONEVALUE)
    except Exception as err:
        raise Exception(red(f"Error during recursive improvement: '{err}'"))
    gr.Warning(whi(f"Recursively improved: {len(prev_prompts)} total examples"))

    whi("Trying to directly embed the new memories")
    prev_prompts = load_prev_prompts(txt_profile).copy()
    to_embed = [content, answer]
    thread = threading.Thread(
        target=embedder,
        kwargs={"text_list": to_embed, "model": shared.pv["choice_embed"]},
        daemon=True,
    )
    thread.start()
    return

@optional_typecheck
@lru_cache
def cached_load_memories(path: str, modtime: float) -> List[dict]:
    "actual code that load the memories, but cached"
    path = Path(path)
    assert path.exists(), f"File not found: {path}"
    content = path.read_text()
    d = json.loads(content)
    assert isinstance(d, list) and all(isinstance(el, dict) for el in d), "Loaded content is not a list of dict"
    return d

@optional_typecheck
def load_prev_prompts(profile: str) -> List[dict]:
    assert Path("profiles/").exists(), "profile directory not found"
    mem_file = Path(f"profiles/{profile}/memories.json")
    if mem_file.exists():
        abs_path = mem_file.resolve().absolute().__str__()
        modtime = mem_file.stat().st_mtime
        prev_prompts=cached_load_memories(path=abs_path, modtime=modtime)

        # Older builds (or hand-edited files) may be missing the "role" key.
        # Backfill in memory so check_prompts() and the few-shot builder don't
        # see a half-formed entry. We don't rewrite the file: the next write
        # from recur_improv() will produce a clean copy on its own.
        for mem in prev_prompts:
            mem.setdefault("role", "user")

        # with open(f"profiles/{profile}/memories.json", "r") as f:
        #     prev_prompts = json.load(f)
        # if Path(f"profiles/{profile}/memories.toml").exists():
        #     try:
        #         with open(f"profiles/{profile}/memories.toml", "r") as f:
        #             prev_prompts_toml = rtoml.load(f)
        #         prev_prompts = check_prompts(prev_prompts)
        #         assert prev_prompts_toml == prev_prompts, "both are different"
        #     except Exception as err:
        #         gr.Warning(red(f"Error when checking toml vs json: '{err}'"))
    else:
        red(f"No memories in profile {profile} found, creating it")
        # Seed the few-shot examples rather than the system prompt: those are the
        # only place the expected card format lives, and both check_prompts() and
        # prompt_filter() reject a role="system" entry since memories.json holds
        # user/assistant example pairs only.
        now = int(time.time())
        prev_prompts = check_prompts([
            {
                "role": "user",
                "content": content,
                "answer": answer,
                "timestamp": now - i,
                "priority": 0,
                "disabled": False,
                "disabled_note": "",
            }
            for i, (content, answer) in enumerate(reversed(SEED_EXAMPLES))
        ], less_verbose=True)
        mem_file.parent.mkdir(parents=True, exist_ok=True)
        with mem_file.open("w") as f:
            json.dump(prev_prompts, f, indent=4, ensure_ascii=False)
        # with open(f"profiles/{profile}/memories.toml", "w") as f:
        #     rtoml.dump(prev_prompts, f, pretty=True, none_value=RTOML_NONEVALUE)

    return prev_prompts


@optional_typecheck
def display_price(sld_max_tkn: int, llm_choice: str) -> str:
    if llm_choice not in shared.llm_info:
        return f"Price not found for model '{llm_choice}'"
    if sld_max_tkn == 0:
        return "Can't set 'Max token' to 0"
    price = shared.llm_info[llm_choice]
    if isinstance(price, float):
        return f"${price} per second (actual price computation is probably wrong for now!)"
    try:
        price_adj = price["input_cost_per_token"] * 0.9 + price["output_cost_per_token"] * 0.1
    except (KeyError, TypeError):
        return f"No token pricing listed for model '{llm_choice}'"
    price_per_request = price_adj * sld_max_tkn
    # free / local models and litellm's placeholder entries all cost 0 per token
    if price_per_request <= 0:
        return f"Model '{llm_choice}' is free or has no listed token pricing"
    price_per_dol = round(1 / price_per_request, 0)
    message = f"Price if all tokens used: ${price_per_request:.5f}."
    message += f"\nRequests per $1: {price_per_dol:.1f} req"
    return message

@optional_typecheck
def get_memories_df(profile: str) -> pd.DataFrame:
    memories = load_prev_prompts(profile).copy()
    if not memories:
        gr.Warning(red(f"No memories found for profile {profile}"))
        return pd.DataFrame()
    for i in range(len(memories)):
        memories[i]["n"] = i + 1
        # role is always present: load_prev_prompts backfills it. .pop keeps
        # the column out of the dataframe the user sees in the Memories tab.
        memories[i].pop("role", None)
    return pd.DataFrame(memories).reset_index().set_index("n")

@optional_typecheck
def get_message_buffer_df() -> pd.DataFrame:
    buffer = shared.message_buffer
    if not buffer:
        gr.Warning(red("No message buffer found"))
        return pd.DataFrame()
    for i in range(len(buffer)):
        buffer[i]["n"] = i + 1
    return pd.DataFrame(buffer).reset_index().set_index("n")

@optional_typecheck
def get_dirload_df() -> pd.DataFrame:
    df = shared.dirload_queue
    # make sure that the index 'n' appears first
    df = df.reset_index().set_index("n").reset_index()
    return df

@optional_typecheck
def split_thinking(prompt: str) -> Tuple[str, str]:
    orig_prompt = copy.deepcopy(prompt)
    thoughts = REG_THINKING.findall(prompt)
    for thought in thoughts:
        prompt = prompt.replace(thought, "", 1)
    prompt = prompt.strip()
    thinking = "\n".join(thoughts).strip()

    state = f"Error when removing thoughts:\nOriginal prompt: '{orig_prompt}'\nThoughts: '{thoughts}'\nOutput prompt: '{prompt}'"
    try:
        assert "<thinking>" not in prompt, state
        assert "</thinking>" not in prompt, state
        assert not re.findall(REG_THINKING, prompt), state
        assert prompt, state
    except Exception:
        gr.Warning(red(state))
        return orig_prompt, ""

    if thinking:
        red("Removed thoughts in prompt")
    return prompt, thinking


