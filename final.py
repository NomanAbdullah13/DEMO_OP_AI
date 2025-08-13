import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CompanyChatBot:
    def __init__(self, website_url):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.website_url = website_url
        self.website_text = self._scrape_website()
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index = None
        self.text_chunks = []
        self._setup_faiss()

    def _scrape_website(self):
        try:
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(self.website_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
                
            # Get text from relevant tags
            texts = soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'article', 'section'])
            content = "\n".join([text.get_text().strip() for text in texts if text.get_text().strip()])
            
            return content[:15000]  # Increased character limit slightly
        except Exception as e:
            st.error(f"Error scraping website: {e}")
            return ""

    def _chunk_text(self, text, chunk_size=500):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def _setup_faiss(self):
        if not self.website_text:
            st.error("No website content available to create embeddings")
            return
            
        # Chunk the website text
        self.text_chunks = self._chunk_text(self.website_text)
        
        # Generate embeddings
        try:
            embeddings = self.embeddings_model.encode(self.text_chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings.astype('float32'))
        except Exception as e:
            st.error(f"Error setting up FAISS index: {e}")

    def _get_relevant_chunks(self, query, k=3):
        if not self.faiss_index:
            return []
            
        try:
            query_embedding = self.embeddings_model.encode([query])
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            return [self.text_chunks[idx] for idx in indices[0] if idx < len(self.text_chunks)]
        except Exception as e:
            st.error(f"Error finding relevant chunks: {e}")
            return []

    def ask_question(self, user_query):
        if not user_query:
            return "Please ask a question."

        if not self.api_key:
            return "OpenAI API key is missing. Please check your configuration."

        try:
            # Get relevant chunks using FAISS
            relevant_chunks = self._get_relevant_chunks(user_query)
            context = "\n".join(relevant_chunks) if relevant_chunks else "No relevant context found."
            
            # Set the OpenAI API key directly
            openai.api_key = self.api_key
            
            # Make the request to the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=350,  # Increased slightly for better responses
                messages=[
                    {"role": "system", "content": """You are a compassionate and empathetic Optimal performance coach assistant. Your role is to:
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
- If user asks direct question not related to company content, respond with "I'm sorry, I can only answer questions related to Optimal Performance System"
- Privacy is important - reassure users their conversations are confidential
- Be patient and allow the conversation to unfold naturally
- Use reflective language ("It sounds like...", "I hear you saying...")
- Balance empathy with gentle challenges to unhelpful thinking patterns"""}, 
                    {"role": "system", "content": f"Company context (use when relevant):\n{context}"},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def run(self):
        st.set_page_config(
            page_title="OP AI", 
            page_icon="🤖",
            layout="centered"
        )
        
        st.title("🤖 DEMO OP AI")
        st.caption("Currently built on content from Optimal Performance System website")
        
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
    # Initialize the chatbot
    chatbot = CompanyChatBot(
        website_url="https://optimalperformancesystem.com/"
    )
    
    # Add a check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY environment variable is not set. Please check your .env file")
    else:
        chatbot.run()
