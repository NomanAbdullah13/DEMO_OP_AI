import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time

# MUST be the very first Streamlit command - DO NOT MOVE THIS
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
        self.client = self._initialize_openai_client()
        self._setup_embeddings()

    def _initialize_embeddings_model(self):
        """Initialize sentence transformer model with error handling"""
        try:
            with st.spinner("🔧 Loading AI model..."):
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading embeddings model: {e}")
            self.embeddings_model = None

    def _initialize_openai_client(self):
        """Initialize OpenAI client with multiple fallback methods"""
        if not self.api_key:
            st.error("🔑 **OpenAI API Key Missing!**")
            st.info("""
            **To fix this:**
            1. Go to your Streamlit Cloud app settings
            2. Click on "Secrets" tab
            3. Add: `OPENAI_API_KEY = "your-api-key-here"`
            4. Save and restart the app
            """)
            return None
        
        # Try multiple initialization methods
        initialization_methods = [
            # Method 1: Minimal initialization
            lambda: OpenAI(api_key=self.api_key),
            
            # Method 2: With explicit base URL
            lambda: OpenAI(
                api_key=self.api_key,
                base_url="https://api.openai.com/v1"
            ),
            
            # Method 3: With timeout but no other params
            lambda: OpenAI(
                api_key=self.api_key,
                timeout=30.0
            ),
            
            # Method 4: Direct HTTP approach (fallback)
            lambda: self._create_direct_client()
        ]
        
        for i, method in enumerate(initialization_methods, 1):
            try:
                client = method()
                # Test the client with a simple call
                try:
                    # Quick test - don't actually call the API, just initialize
                    if hasattr(client, 'api_key'):
                        return client
                except:
                    pass
                return client
            except Exception as e:
                if i == len(initialization_methods):
                    st.error(f"❌ All OpenAI client initialization methods failed. Last error: {e}")
                    return None
                continue
        
        return None
    
    def _create_direct_client(self):
        """Fallback: Create client with minimal configuration"""
        import openai
        # Set the API key directly (older method)
        openai.api_key = self.api_key
        return OpenAI(api_key=self.api_key)

    def _scrape_website(self):
        """Scrape website content with robust error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            with st.spinner("🌐 Loading website content..."):
                response = requests.get(self.website_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer"]):
                    script.decompose()
                
                # Extract relevant text content
                texts = soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'div'])
                content = []
                
                for text in texts:
                    text_content = text.get_text().strip()
                    if text_content and len(text_content) > 20:  # Filter out short/empty texts
                        content.append(text_content)
                
                full_content = "\n".join(content)
                
                # Limit content size for processing
                if len(full_content) > 15000:
                    full_content = full_content[:15000]
                
                return full_content if full_content else "Optimal Performance coaching services and resources available."
                
        except requests.RequestException as e:
            st.warning(f"⚠️ Could not load website content: {e}")
            return "Website content could not be loaded. I can still help with general performance coaching questions."
        except Exception as e:
            st.warning(f"⚠️ Error processing website: {e}")
            return "I can help you with performance coaching questions based on general knowledge."
    
    def _chunk_text(self, text, chunk_size=400):
        """Split text into manageable chunks"""
        if not text or len(text.strip()) == 0:
            return ["General performance coaching information available."]
        
        # Split by sentences first, then by words if needed
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
        
        return chunks if chunks else ["General performance coaching information available."]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0
    
    def _setup_embeddings(self):
        """Setup embeddings for semantic search"""
        if not self.embeddings_model:
            st.error("❌ Embeddings model not available")
            return
            
        try:
            with st.spinner("🔍 Processing content for intelligent responses..."):
                # Chunk the website text
                self.text_chunks = self._chunk_text(self.website_text)
                
                # Generate embeddings
                self.chunk_embeddings = self.embeddings_model.encode(
                    self.text_chunks, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                st.success(f"✅ Successfully processed {len(self.text_chunks)} content sections")
                
        except Exception as e:
            st.warning(f"⚠️ Error processing content: {e}")
            # Fallback
            self.text_chunks = ["General performance coaching information available."]
            try:
                self.chunk_embeddings = self.embeddings_model.encode(self.text_chunks)
            except:
                self.chunk_embeddings = np.array([[0.0] * 384])  # Dummy embedding

    def _get_relevant_chunks(self, query, k=3):
        """Get most relevant content chunks for the query"""
        if not self.embeddings_model or len(self.chunk_embeddings) == 0:
            return self.text_chunks[:k] if self.text_chunks else ["General coaching guidance available."]
            
        try:
            # Encode the query
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)[0]
            
            # Calculate similarities
            similarities = []
            for chunk_embedding in self.chunk_embeddings:
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                similarities.append(similarity)
            
            # Get top k most relevant chunks
            if similarities:
                top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
                return [self.text_chunks[idx] for idx in top_indices]
            else:
                return self.text_chunks[:k]
                
        except Exception as e:
            st.warning(f"⚠️ Error retrieving relevant content: {e}")
            return self.text_chunks[:k]

    def ask_question(self, user_query):
        """Process user question and generate AI response"""
        if not user_query or not user_query.strip():
            return "Please ask a question, and I'll be happy to help you! 😊"

        if not self.client:
            return "❌ I'm having trouble connecting to the AI service. Please check if the OpenAI API key is properly configured in the app settings."

        try:
            # Get relevant content chunks
            relevant_chunks = self._get_relevant_chunks(user_query, k=3)
            context = "\n\n".join(relevant_chunks)
            
            # Generate AI response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  
                max_tokens=350,
                temperature=0.7,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a compassionate and empathetic Optimal Performance coach assistant. Your mission is to help people unlock their potential and achieve peak performance.

🎯 **Your Core Approach:**
- **Active Listening**: Validate feelings and truly understand their situation
- **Thoughtful Inquiry**: Ask open-ended questions that promote self-discovery
- **Empowering Guidance**: Help them find their own solutions rather than prescribing answers
- **Trust Building**: Maintain warmth, authenticity, and genuine care
- **Growth Mindset**: Gently challenge limiting beliefs and negative patterns
- **Opportunity Focus**: Help them see challenges as growth opportunities

💭 **For Emotional/Mental Concerns:**
- Acknowledge their struggle without immediately trying to "fix" it
- Create space for them to explore and express their feelings
- Normalize their experiences when appropriate
- Guide them toward self-reflection and personal insights
- Ask: "What does this situation want to teach you?"

🚀 **For Performance/Goal-Related Questions:**
- Focus on process improvement over outcome obsession
- Break down big goals into small, actionable steps
- Encourage experimentation and learning from setbacks
- Connect their efforts to deeper purpose and values
- Ask: "What would taking one small step forward look like?"

📋 **Response Guidelines:**
- Keep responses conversational and avoid being preachy
- Use reflective language: "It sounds like...", "I hear you saying...", "Help me understand..."
- Balance empathy with gentle accountability
- If questions are completely unrelated to coaching/performance, politely redirect
- Always maintain confidentiality and create a safe space
- End with a thoughtful question when appropriate

Remember: You're not just giving advice - you're facilitating their own wisdom and growth. 🌱"""
                    },
                    {
                        "role": "system", 
                        "content": f"**Relevant Context from Company/Website** (use when applicable):\n{context}"
                    },
                    {
                        "role": "user", 
                        "content": user_query
                    }
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                return "⏰ I'm receiving many questions right now. Please wait a moment and try again."
            elif "quota" in error_msg or "billing" in error_msg:
                return "💳 The AI service is temporarily unavailable due to quota limits. Please try again later or contact support."
            elif "api key" in error_msg or "authentication" in error_msg:
                return "🔑 There's an authentication issue with the API key. Please contact the administrator."
            elif "timeout" in error_msg:
                return "⏱️ The request timed out. Please try asking your question again."
            else:
                return "⚠️ I'm experiencing technical difficulties right now. Please try again in a moment."
    
    def run(self):
        """Main application interface"""
        
        # Custom CSS for better styling
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
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
        }
        .sidebar-content {
            background: #F8F9FA;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .feature-box {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main Header
        st.markdown("""
        <div class="main-header">🤖 OP AI</div>
        <div class="sub-header">Your Optimal Performance Coach</div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Check if everything is properly initialized
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
        
        # Chat Interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat Input
        if prompt := st.chat_input("💬 Share what's on your mind...", key="user_input"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking deeply about your question..."):
                    response = self.ask_question(prompt)
                st.markdown(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 🌟 About OP AI")
            st.markdown("""
            <div class="sidebar-content">
            Your AI-powered performance coach designed to help you:
            <br><br>
            🎯 <strong>Achieve Goals</strong> - Strategic planning & execution<br>
            💪 <strong>Build Resilience</strong> - Overcome setbacks & challenges<br>
            🧠 <strong>Growth Mindset</strong> - Transform limiting beliefs<br>
            🤝 <strong>Emotional Support</strong> - Navigate difficult emotions<br>
            ⚡ <strong>Peak Performance</strong> - Optimize your potential
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔒 Privacy & Confidentiality")
            st.info("Your conversations are private and secure. Nothing is stored permanently.")
            
            st.markdown("### 🛠️ Chat Management")
            if st.button("🗑️ Clear Chat History", help="Start a fresh conversation"):
                st.session_state.messages = [
                    {
                        "role": "assistant", 
                        "content": "Hello again! I'm ready for a fresh conversation. What would you like to explore today? 😊"
                    }
                ]
                st.rerun()
            
            st.markdown("### 💡 Tips for Better Conversations")
            st.markdown("""
            <div class="sidebar-content">
            • <strong>Be specific</strong> about your situation<br>
            • <strong>Share context</strong> - what's really going on?<br>
            • <strong>Ask follow-up questions</strong><br>
            • <strong>Be open</strong> to exploring new perspectives<br>
            • <strong>Take your time</strong> - there's no rush
            </div>
            """, unsafe_allow_html=True)
            
            # Debug info for development
            if st.checkbox("🔧 Debug Info", help="Show technical details"):
                st.markdown("**System Status:**")
                st.write(f"✅ API Key: {'Configured' if self.api_key else 'Missing'}")
                st.write(f"✅ Website Content: {len(self.website_text)} characters")
                st.write(f"✅ Content Chunks: {len(self.text_chunks)}")
                st.write(f"✅ Embeddings: {'Ready' if len(self.chunk_embeddings) > 0 else 'Not Ready'}")
                st.write(f"✅ OpenAI Client: {'Connected' if self.client else 'Error'}")

# Main Application Entry Point
def main():
    """Initialize and run the OP AI application"""
    try:
        # Initialize chatbot
        chatbot = CompanyChatBot(
            website_url="https://optimalperformancesystem.com/"
        )
        
        # Run the application
        chatbot.run()
        
    except Exception as e:
        st.error(f"❌ **Application Error**: {e}")
        st.info("Please refresh the page. If the problem persists, contact support.")
        
        # Show detailed error for debugging
        with st.expander("🔧 Technical Details", expanded=False):
            st.code(str(e))

# Application Entry Point
if __name__ == "__main__":
    main()
