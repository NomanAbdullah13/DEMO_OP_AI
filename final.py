import streamlit as st
from openai import OpenAI
import openai
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

class CompanyChatBot:
    def __init__(self, website_url):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.website_url = website_url
        self.website_text = self._scrape_website()
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index = None
        self.text_chunks = []
        openai.api_key = self.api_key
        self._setup_faiss()

    def _scrape_website(self):
        try:
            response = requests.get(self.website_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            texts = soup.find_all(['p', 'li', 'h1', 'h2', 'h3'])
            content = "\n".join([text.get_text() for text in texts])
            return content[:12000]
        except Exception as e:
            return f"Error scraping website: {e}"
    
    def _chunk_text(self, text, chunk_size=500):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    def _setup_faiss(self):
        # Chunk the website text
        self.text_chunks = self._chunk_text(self.website_text)
        
        # Generate embeddings
        embeddings = self.embeddings_model.encode(self.text_chunks)
            
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))

    def _get_relevant_chunks(self, query, k=3):
        query_embedding = self.embeddings_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        return [self.text_chunks[idx] for idx in indices[0]]

    def ask_question(self, user_query):
        if not user_query:
            return "Please ask a question."

        try:
            # Get relevant chunks using FAISS
            relevant_chunks = self._get_relevant_chunks(user_query)
            context = "\n".join(relevant_chunks)
            
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  
                max_tokens=250,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a compassionate and empathetic Optimal performance coach assistant. Your role is to:
1. Actively listen and validate the user's feelings
2. Ask thoughtful, open-ended questions to understand their situation deeply
3. Provide supportive guidance while helping them find their own solutions
4. Maintain a warm, conversational tone that builds trust
5. When appropriate, gently challenge negative thought patterns
6. Help users connect their experiences to potential growth opportunities

For emotional concerns:
- Acknowledge the difficulty without immediately trying to fix it
- Help the user explore their feelings and experiences
- Normalize struggles when appropriate
- Guide them toward self-reflection and personal insights

For performance-related questions:
- Focus on process over outcomes
- Help identify small, actionable steps
- Encourage a growth mindset
- Connect to relevant company resources when applicable

Remember:
- If user asks direct question not related to company content, respond with I cannot answer that or something like that"
- Privacy is important - reassure users their conversations are confidential
- Be patient and allow the conversation to unfold naturally
- Use reflective language ("It sounds like...", "I hear you saying...")
- Balance empathy with gentle challenges to unhelpful thinking patterns"""
                    },
                    {"role": "system", "content": f"Company context (use when relevant):\n{context}"},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def run(self):
        st.set_page_config(page_title="OP AI", layout="centered")
        st.title("🤖 DEMO OP AI(Currently build on your website. After giving ) ")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Text input
        user_query = st.chat_input("How can I support you today?")
        
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Get assistant response
            with st.spinner("Thinking..."):
                assistant_response = self.ask_question(user_query)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

if __name__ == "__main__":
    chatbot = CompanyChatBot(
        website_url="https://optimalperformancesystem.com/"
    )
    chatbot.run()
