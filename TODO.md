# Voice2Anki Roadmap

This file is the destination for [MdXLogseqTODOSync](https://github.com/thiswillbeyourgithub/MdXLogseqTODOSync). Items between the `<!-- BEGIN_TODO -->` / `<!-- END_TODO -->` markers are overwritten by the sync tool — edit them in your Logseq source page, not here.

<!-- BEGIN_TODO -->
- ### Urgent
- add a tab with the documentation inside
    - starts from the problem
    - then explain why V2A exists
    - then introduce features naturally
- switch to gradio 5
    - https://github.com/gradio-app/gradio/issues/9463
- make easy to install via pypi / stop using requirements
- Make the `utils/cli.py` version a first class citizen
- ### Enhancements
- by default create a Voice2Anki deck inside anki if needed
- create a project icon
- display the total price in the settings
- API should be set as a textbox that works for all APIs instead of a dedicated field each time
- use the litellm tokenizer as it's a bit better than relying only on openai
- the system prompt should contain a string like {EXTRA_RULES} so that the user can add its own rules
- add  a setting for a list of tags that you can add with a quick button to the previous card
- use pandas to handle the embeddings instead of lists. This probably makes the score computation non scalable.
- in the prompt make the LLM use a <thinking> xml tag
- make it easier to change the endpoint url for whisper
- checkbox to disable OCR + to set the OCR language
- for each prompt used, keep a counter of how many times it is used, and a counter of how many times it is used on the same audio inputs
    - as if it's used say 10 times on the same prompt, that means it was not sufficient and might be a bad example
    - so the examples that have the highest ratio failed/used should be manually vetted
- replace most hardcoded strings by variables in a py file
- store the thoughts in the memories maybe?
- ### Overhaul
- use faiss (possibly langchain) to handle the embeddings as it currently might not be scalable.
- change the way audo components are used
    - create like 1000 components
    - using a sliding window: render them, display them, unrender them etc. probably using gr.update
    - use the shared class to handle the window
    - modify the audio events so that they send/receive the first or last only
- convert more of the code to use async
- add a column to add buttons to easily add a text to the prompt or the audio. This way, modifications that the user frequently has to do are quicker to do.
<!-- END_TODO -->