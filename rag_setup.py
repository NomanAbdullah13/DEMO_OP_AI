import yaml
from langchain.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load sources from YAML
with open('data/sources.yaml', 'r') as f:
    sources = yaml.safe_load(f)['sources']

docs = []
for s in sources:
    loader = WebBaseLoader(s['url'])
    raw_docs = loader.load()
    for d in raw_docs:
        d.metadata = {
            'tier': s['tier'],
            'source_id': s['source_id'],
            'age_applicability': s['age_applicability'],
            'pub_year': s['pub_year'],
            'doc_type': s['doc_type']
        }
    docs.extend(raw_docs)

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# Load OP corpus (Tier OP)
op_loader = DirectoryLoader('data/op_corpus/', glob="**/*.txt", loader_cls=TextLoader)
op_docs = op_loader.load()
for d in op_docs:
    d.metadata = {'tier': 'OP', 'source_id': 'OP', 'age_applicability': 'all', 'pub_year': 2024, 'doc_type': 'brand_content'}
split_op = splitter.split_documents(op_docs)
split_docs.extend(split_op)

# Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
vectorstore.persist()
print("Vector store built successfully.")