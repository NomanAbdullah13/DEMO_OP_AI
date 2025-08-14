import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="OP AI Coach", 
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

class CompanyChatBot:
    def __init__(self, website_url):
        # Get API key from Streamlit secrets or environment
        self.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.website_url = website_url
        self.website_text = self._scrape_website()
        self.embeddings_model = None
        self.text_chunks = []
        self.chunk_embeddings = []
        # Initialize components
        self._initialize_embeddings_model()
        self._setup_embeddings()

    def _initialize_embeddings_model(self):
        """Initialize sentence transformer model"""
        try:
            with st.spinner("🔧 Loading AI model..."):
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading embeddings model: {e}")
            self.embeddings_model = None

    def _scrape_website(self):
        """Scrape website content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            with st.spinner("🌐 Loading website content..."):
                response = requests.get(self.website_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for script in soup(["script", "style", "nav", "footer"]):
                    script.decompose()
                
                # Extract text
                texts = soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'div'])
                content = []
                
                for text in texts:
                    text_content = text.get_text().strip()
                    if text_content and len(text_content) > 20:
                        content.append(text_content)
                
                full_content = "\n".join(content)
                
                if len(full_content) > 15000:
                    full_content = full_content[:15000]
                
                return full_content if full_content else "Optimal Performance coaching services available."
                
        except Exception as e:
            st.warning(f"⚠️ Could not load website content: {e}")
            return "I can help with general performance coaching questions."
    
    def _chunk_text(self, text, chunk_size=400):
        """Split text into chunks"""
        if not text or len(text.strip()) == 0:
            return ["General performance coaching information available."]
        
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks if chunks else ["General coaching information available."]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
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
        """Setup embeddings"""
        if not self.embeddings_model:
            st.error("❌ Embeddings model not available")
            return
            
        try:
            with st.spinner("🔍 Processing content..."):
                self.text_chunks = self._chunk_text(self.website_text)
                self.chunk_embeddings = self.embeddings_model.encode(
                    self.text_chunks, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                st.success(f"✅ Successfully processed {len(self.text_chunks)} content sections")
                
        except Exception as e:
            st.warning(f"⚠️ Error processing content: {e}")
            self.text_chunks = ["General coaching information available."]
            try:
                self.chunk_embeddings = self.embeddings_model.encode(self.text_chunks)
            except:
                self.chunk_embeddings = np.array([[0.0] * 384])

    def _get_relevant_chunks(self, query, k=3):
        """Get relevant content chunks"""
        if not self.embeddings_model or len(self.chunk_embeddings) == 0:
            return self.text_chunks[:k] if self.text_chunks else ["General coaching available."]
            
        try:
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)[0]
            
            similarities = []
            for chunk_embedding in self.chunk_embeddings:
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                similarities.append(similarity)
            
            if similarities:
                top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
                return [self.text_chunks[idx] for idx in top_indices]
            else:
                return self.text_chunks[:k]
                
        except Exception as e:
            return self.text_chunks[:k]

    def _call_openai_direct(self, messages, max_tokens=350, temperature=0.7):
        """Direct HTTP call to OpenAI API (proxy-free)"""
        if not self.api_key:
            return "API key not configured."
        
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            if "rate limit" in str(e).lower():
                return "⏰ Rate limit reached. Please wait a moment and try again."
            elif "quota" in str(e).lower():
                return "💳 API quota exceeded. Please check your OpenAI account."
            else:
                return f"⚠️ API request failed: {str(e)[:100]}..."
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)[:100]}..."

    def ask_question(self, user_query):
        """Process user question"""
        if not user_query or not user_query.strip():
            return "Please ask a question, and I'll be happy to help! 😊"

        if not self.api_key:
            return "❌ OpenAI API key is not configured. Please check the app settings."

        try:
            # Get relevant content
            relevant_chunks = self._get_relevant_chunks(user_query, k=3)
            context = "\n\n".join(relevant_chunks)
            
            messages = [
                {
                    "role": "system", 
                    "content": """You are a compassionate and empathetic Optimal Performance coach assistant. Your mission is to help people unlock their potential and achieve peak performance.

🎯 **Your Core Approach:**
1. Actively listen and validate the user's feelings
2. Ask thoughtful, open-ended questions to understand their situation deeply
3. Provide supportive guidance while helping them find their own solutions
4. Maintain a warm, conversational tone that builds trust
5. When appropriate, gently challenge negative thought patterns
6. Help users connect their experiences to potential growth opportunities

💭 **For Emotional/Mental Concerns:**
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
                {
                    "role": "system", 
                    "content": f"**Context from Company/Website** (use when relevant):\n{context}"
                },
                {
                    "role": "user", 
                    "content": user_query
                }
            ]
            
            return self._call_openai_direct(messages)
            
        except Exception as e:
            return f"⚠️ I'm experiencing technical difficulties: {str(e)[:100]}..."
    
    def run(self):
        """Main application interface"""
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #2E86C1;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .sub-header {
            text-align: center;
            color: #566573;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">🤖 OP AI</div>
        <div class="sub-header">Your Optimal Performance Coach</div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Check API key
        if not self.api_key:
            st.error("🔑 **Configuration Required**")
            st.info("""
            **To activate OP AI:**
            1. Go to your Streamlit Cloud app settings ⚙️
            2. Navigate to "Secrets" tab 🔐
            3. Add: `OPENAI_API_KEY = "your-openai-api-key"`
            4. Save and restart the app 🔄
            """)
            st.stop()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": """👋 **Welcome to OP AI!** 

I'm your Optimal Performance Coach, here to support you on your journey to peak performance and personal growth.

**I can help you with:**
🎯 Goal setting and achievement strategies
💪 Building resilience and confidence
🧠 Developing a growth mindset
⚡ Overcoming performance barriers
🌱 Personal development and self-improvement

**What's on your mind today?** Share anything you'd like to explore - challenges you're facing, goals you want to achieve, or areas where you'd like to grow. I'm here to listen and guide you toward your own insights and solutions. 😊"""
                }
            ]
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💬 Share what's on your mind...", key="user_input"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = self.ask_question(prompt)
                st.markdown(response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 🌟 About OP AI")
            st.markdown("""
            Your AI-powered performance coach designed to help you:
            
            🎯 **Achieve Goals** - Strategic planning & execution  
            💪 **Build Resilience** - Overcome challenges  
            🧠 **Growth Mindset** - Transform limiting beliefs  
            🤝 **Emotional Support** - Navigate difficulties  
            ⚡ **Peak Performance** - Optimize your potential
            """)
            
            st.markdown("### 🔒 Privacy")
            st.info("Your conversations are private and secure.")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = [
                    {
                        "role": "assistant", 
                        "content": "Hello again! Ready for a fresh conversation. What would you like to explore? 😊"
                    }
                ]
                st.rerun()

# Main function
def main():
    """Initialize and run the application"""
    try:
        chatbot = CompanyChatBot(
            website_url="https://optimalperformancesystem.com/"
        )
        chatbot.run()
        
    except Exception as e:
        st.error(f"❌ **Application Error**: {e}")
        st.info("Please refresh the page or contact support.")

if __name__ == "__main__":
    main()


