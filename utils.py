import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model="gpt-4o", temperature=0.6)

def classify_query(query):
    template = ChatPromptTemplate.from_template(classification_prompt + "\nQuery: {query}")
    chain = LLMChain(llm=llm, prompt=template)
    output = chain.predict(query=query)
    return yaml.safe_load(output)

