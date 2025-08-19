import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import PubMedRetriever
from prompts import system_prompt
from utils import classify_query

load_dotenv()
print("API Key:", os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
)
embeddings = OpenAIEmbeddings(
    
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
pubmed_retriever = PubMedRetriever()

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

template = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)
chain = LLMChain(llm=llm, prompt=template, memory=memory)

print("OP AI: Hello! I'm here to help. What's your age group (youth <=17, adult 18-39, masters 40+)? And what's on your mind?")
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Classify
    classification = classify_query(user_input)
    risk = classification.get('risk', 'low')
    age_group = classification.get('age_group', 'unknown')
    # Assume age detected or asked; fallback to adult if unknown

    if risk == 'high':
        allowed_tiers = [1]
    elif risk == 'medium':
        allowed_tiers = [1, 2]
    else:
        allowed_tiers = [1, 2, 3, 'OP']

    # Retrieve
    retrieved = []
    # Tier 1 from vectorstore
    docs = vectorstore.similarity_search(user_input, k=5, filter={'tier': 1})
    if docs:
        retrieved.extend(docs)

    # Fallback to Tier 2 if needed
    if len(retrieved) < 3 and 2 in allowed_tiers:
        pub_docs = pubmed_retriever.invoke(user_input, top_k=3)
        for d in pub_docs:
            d.metadata['tier'] = 2
            d.metadata['source_id'] = 'PubMed'
        retrieved.extend(pub_docs)

    # Add OP if low risk
    if risk == 'low' and 'OP' in allowed_tiers:
        op_docs = vectorstore.similarity_search(user_input, k=2, filter={'tier': 'OP'})
        retrieved.extend(op_docs)

    # Format context
    context = "\n".join([f"Source: {d.metadata['source_id']} (Tier {d.metadata['tier']})\n{d.page_content}\n" for d in retrieved])

    # Augment input
    augmented_input = f"Context:\n{context}\n\nUser: {user_input}"

    # Generate response
    response = chain.predict(input=augmented_input)
    print("App:", response)

