from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = os.getenv("model")
    
    
llm = HuggingFaceEndpoint(
    model=model,
    huggingfacehub_api_token=os.getenv("api_key"),
)

chat = ChatHuggingFace(llm = llm)
    
    
temp = """ Analyse this image and tell me the details. 
    Image : {image_url}
    Report :
    """
prompt = PromptTemplate(template=temp, input_variables=["image_url"])

chain = prompt | chat

def analyse_image(image_url):
    resp = chain.invoke({'image_url' : image_url})
    return resp


imageurl = "https://www.google.com/imgres?q=chest%20cancer%20x%20ray&imgurl=https%3A%2F%2Fradiologybusiness.com%2Fsites%2Fdefault%2Ffiles%2Fassets%2Farticles%2F4996132.jpg&imgrefurl=https%3A%2F%2Fradiologybusiness.com%2Ftopics%2Fmedical-imaging%2Fradiography%2Fcommon-features-missed-lung-cancer-chest-x-ray&docid=b0Azw3zW75HOxM&tbnid=92YDhTVJbkCu6M&vet=12ahUKEwjm34HmwMGOAxUwxDgGHVdbH5oQM3oECBQQAA..i&w=800&h=656&hcb=2&ved=2ahUKEwjm34HmwMGOAxUwxDgGHVdbH5oQM3oECBQQAA"

ai_analysis = analyse_image(imageurl)
ai_analysis.pretty_print()







