import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class CompanyChatBot:
    def __init__(self, website_url):
        # Get API key from Streamlit secrets or environment
        self.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.website_url = website_url
        self.website_text = self._scrape_website()
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_chunks = []
        self.chunk_embeddings = []
        # Initialize OpenAI client properly
        self.client = self._initialize_openai_client()
        self._setup_embeddings()

    def _initialize_openai_client(self):
        """Initialize OpenAI client with error handling"""
        try:
            if not self.api_key:
                st.error("🔑 OpenAI API key is missing!")
                return None
            client = OpenAI(api_key=self.api_key)
            return client
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {e}")
            return None

    def _scrape_website(self):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.website_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            texts = soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'div', 'span'])
            content = "\n".join([text.get_text().strip() for text in texts if text.get_text().strip()])
            
            if len(content) > 15000:
                content = content[:15000]
            
            return content if content else "Default company information available."
            
        except Exception as e:
            st.warning(f"Could not load website content: {e}")
            return "Website content could not be loaded. I can still help with general coaching questions."
    
    def _chunk_text(self, text, chunk_size=400):
        if not text or len(text.strip()) == 0:
            return ["No content available"]
        
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks if chunks else ["No content available"]
    
    def _cosine_similarity(self, vec1, vec2):
        """Simple cosine similarity calculation"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
        except:
            return 0
    
    def _setup_embeddings(self):
        """Setup embeddings using simple numpy operations"""
        try:
            with st.spinner("🔍 Processing website content..."):
                # Chunk the website text
                self.text_chunks = self._chunk_text(self.website_text)
                
                # Generate embeddings
                self.chunk_embeddings = self.embeddings_model.encode(self.text_chunks)
                st.success(f"✅ Successfully processed {len(self.text_chunks)} content sections")
                
        except Exception as e:
            st.warning(f"Error processing content: {e}")
            # Fallback
            self.text_chunks = ["General company information available"]
            self.chunk_embeddings = self.embeddings_model.encode(self.text_chunks)

    def _get_relevant_chunks(self, query, k=3):
        """Get relevant chunks using simple similarity calculation"""
        try:
            # Encode the query
            query_embedding = self.embeddings_model.encode([query])[0]
            
            # Calculate similarities manually
            similarities = []
            for chunk_embedding in self.chunk_embeddings:
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                similarities.append(similarity)
            
            # Get top k chunks
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
            
            return [self.text_chunks[idx] for idx in top_indices]
        except Exception as e:
            st.warning(f"Error retrieving relevant content: {e}")
            return self.text_chunks[:3]

    def ask_question(self, user_query):
        if not user_query:
            return "Please ask a question."

        if not self.client:
            return "Sorry, I'm having trouble connecting to the AI service. Please check the API key configuration."

        try:
            # Get relevant chunks
            relevant_chunks = self._get_relevant_chunks(user_query)
            context = "\n".join(relevant_chunks)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  
                max_tokens=300,
                temperature=0.7,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a compassionate and empathetic Optimal Performance coach assistant. Your role is to:

1. 🎯 **Active Listening**: Validate the user's feelings and experiences
2. 🤔 **Thoughtful Questions**: Ask open-ended questions to understand their situation deeply  
3. 💪 **Supportive Guidance**: Help them find their own solutions rather than giving direct advice
4. 🤝 **Trust Building**: Maintain a warm, conversational tone
5. 🧠 **Growth Mindset**: Gently challenge negative thought patterns when appropriate
6. 🌱 **Opportunity Focus**: Help connect experiences to potential growth

**For Emotional Concerns:**
- Acknowledge difficulty without immediately trying to fix it
- Help explore feelings and experiences
- Normalize struggles appropriately
- Guide toward self-reflection and personal insights

**For Performance Questions:**
- Focus on process over outcomes
- Identify small, actionable steps
- Encourage growth mindset
- Connect to relevant resources when applicable

**Guidelines:**
- If questions aren't related to coaching/company content, politely redirect
- Reassure about conversation confidentiality
- Use reflective language ("It sounds like...", "I hear you saying...")
- Balance empathy with gentle challenges to unhelpful thinking"""
                    },
                    {"role": "system", "content": f"Company/Website Context (use when relevant):\n{context}"},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                return "⏰ I'm receiving many questions right now. Please wait a moment and try again."
            elif "quota" in error_msg or "billing" in error_msg:
                return "💳 The AI service is temporarily unavailable due to quota limits. Please try again later."
            elif "api key" in error_msg:
                return "🔑 There's an authentication issue with the API key. Please contact support."
            else:
                return f"⚠️ I'm having trouble processing your question: {str(e)[:100]}... Please try again."
    
    def run(self):
        # Page configuration
        st.set_page_config(
            page_title="OP AI Coach", 
            page_icon="🤖",
            layout="centered",
            initial_sidebar_state="collapsed"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
        }
        .stChatMessage {
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🤖 OP AI - Optimal Performance Coach</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Check API key
        if not self.api_key:
            st.error("🔑 **OpenAI API Key Missing!**")
            st.info("""
            **For Streamlit Cloud Deployment:**
            1. Go to your app settings
            2. Go to "Secrets" section
            3. Add: `OPENAI_API_KEY = "your-api-key-here"`
            """)
            st.stop()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your Optimal Performance Coach. How can I support you today? Whether you're looking to improve performance, work through challenges, or explore growth opportunities, I'm here to help. 😊"}
            ]
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Share what's on your mind..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = self.ask_question(prompt)
                st.markdown(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Sidebar with info
        with st.sidebar:
            st.markdown("### 💡 About OP AI")
            st.markdown("""
            Your personal Optimal Performance Coach designed to:
            - 🎯 Help you achieve your goals
            - 💪 Build resilience and confidence  
            - 🧠 Develop growth mindset
            - 🤝 Provide empathetic support
            """)
            
            st.markdown("### 🔒 Privacy")
            st.info("Your conversations are confidential and not stored permanently.")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm your Optimal Performance Coach. How can I support you today? 😊"}
                ]
                st.rerun()

# Main execution
def main():
    try:
        chatbot = CompanyChatBot(
            website_url="https://optimalperformancesystem.com/"
        )
        chatbot.run()
    except Exception as e:
        st.error(f"❌ Failed to start the chatbot: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
