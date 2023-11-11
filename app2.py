# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('Mequity AI')
prompt = st.text_input('What can we help you with?') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Answer my medical and vaccine questions about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research','medical_history'], 
    template='Here is my medical history: {medical_history} answer my medical question: {title} while using important stats from this research:{wikipedia_research}'

)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.2) # Temperature
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
wiki = WikipediaAPIWrapper()

# Load the patients medical history
medical_history = "Patient is taking penacilin and has a history of hypertension and diabetes."

# Show stuff to the screen if there's a prompt
if prompt: 
    if 'appointment' in prompt.lower():
        resources = "Here are some resources to make an appointment: https://doctorswithafrica.org/, https://www.google.com/search?q=find+doctors+in+africa&sca_esv=581565025&sxsrf=AM9HkKn5SC9ek2fZovfMEm02Vo57l0ZUQA%3A1699732445236&ei=3dtPZcP4DeOm5NoPiaix-Aw&ved=0ahUKEwiD2_Hk3LyCAxVjE1kFHQlUDM8Q4dUDCBE&uact=5&oq=find+doctors+in+africa&gs_lp=Egxnd3Mtd2l6LXNlcnAaAhgCIhZmaW5kIGRvY3RvcnMgaW4gYWZyaWNhMgUQIRigATIIECEYFhgeGB0yCBAhGBYYHhgdMggQIRgWGB4YHTIIECEYFhgeGB0yCBAhGBYYHhgdSItNUMQCWPdLcAx4ApABAJgBngGgAc4ZqgEFMjAuMTO4AQPIAQD4AQGoAhTCAgQQABhHwgIHECMY6gIYJ8ICGRAAGAMYjwEY5QIY6gIYtAIYjAMYiwPYAQHCAgQQIxgnwgIHECMYigUYJ8ICCBAAGIoFGJECwgIREC4YgAQYsQMYgwEYxwEY0QPCAgsQABiABBixAxiDAcICFBAuGIoFGLEDGIMBGMcBGNEDGJECwgIKEC4YgAQYFBiHAsICCxAAGIoFGLEDGJECwgIFEAAYgATCAg0QABiABBgUGIcCGLEDwgIOEC4YgAQYsQMYgwEY1ALCAhAQABiABBgUGIcCGLEDGIMBwgIIEAAYgAQYsQPCAgYQABgWGB7iAwQYACBBiAYBkAYIugYGCAEQARgL&sclient=gws-wiz-serp"
        st.write(resources)


    response = llm(f'Use my medical history: {medical_history} to answer my medical questions: {prompt} taking into account my history while using important stats from research')
    st.write(response)
    
    # Print the wiki research
    wiki_research = wiki.run(prompt) 
    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)