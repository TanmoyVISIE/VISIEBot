import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class PostgreSQLHistoryManager:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }
        
        # Validate connection parameters
        print(f"Connecting to PostgreSQL with:")
        print(f"Host: {self.connection_params['host']}")
        print(f"Database: {self.connection_params['database']}")
        print(f"User: {self.connection_params['user']}")
        print(f"Port: {self.connection_params['port']}")
        
        self.init_database()
        # Don't create session on init
        self.current_session_id = None

    def get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.connection_params)
        except psycopg2.OperationalError as e:
            print(f"Database connection error: {e}")
            print("Please ensure:")
            print("1. PostgreSQL is running")
            print("2. Database exists")
            print("3. User credentials are correct")
            print("4. .env file has correct POSTGRES_* variables")
            raise

    def init_database(self):
        """Initialize the chat history table"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) UNIQUE NOT NULL,
                        messages JSONB DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster session lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON chat_history(session_id)
                """)
                
                conn.commit()
    
    def create_new_session(self) -> str:
        """Create a new chat session and return session ID"""
        session_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_history (session_id, messages)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                """, (session_id, json.dumps([])))
                conn.commit()
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get current messages
                cursor.execute("""
                    SELECT messages FROM chat_history 
                    WHERE session_id = %s
                """, (session_id,))
                
                result = cursor.fetchone()
                if result:
                    current_messages = result[0] if result[0] else []
                    current_messages.append(message)
                    
                    # Update with new message
                    cursor.execute("""
                        UPDATE chat_history 
                        SET messages = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE session_id = %s
                    """, (json.dumps(current_messages), session_id))
                else:
                    # Create new session if it doesn't exist
                    cursor.execute("""
                        INSERT INTO chat_history (session_id, messages)
                        VALUES (%s, %s)
                    """, (session_id, json.dumps([message])))
                
                conn.commit()
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get all messages for a session"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT messages FROM chat_history 
                    WHERE session_id = %s
                """, (session_id,))
                
                result = cursor.fetchone()
                if result and result['messages']:
                    return result['messages']
                return []
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 1 FROM chat_history 
                    WHERE session_id = %s
                """, (session_id,))
                
                return cursor.fetchone() is not None
    
    def clear_session(self, session_id: str):
        """Clear all messages for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE chat_history 
                    SET messages = '[]', updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                """, (session_id,))
                conn.commit()
    
    def delete_session(self, session_id: str):
        """Delete a session completely"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM chat_history 
                    WHERE session_id = %s
                """, (session_id,))
                conn.commit()
    
    def get_formatted_history_for_llm(self, session_id: str) -> str:
        """Get formatted history for LLM context"""
        messages = self.get_session_history(session_id)
        
        if not messages:
            return ""
        
        formatted_history = []
        for msg in messages[-10:]:  # Only last 10 messages to avoid token limits
            role = msg['role']
            content = msg['content']
            if role == 'user':
                formatted_history.append(f"User: {content}")
            elif role == 'assistant':
                formatted_history.append(f"Assistant: {content}")
        
        return "\n".join(formatted_history)
    
    def get_current_session(self) -> str:
        """Get the current active session, create if none exists"""
        if not self.current_session_id:
            self.current_session_id = self.create_new_session()
            print(f"New session created on first request: {self.current_session_id}")
        return self.current_session_id
    
    def start_new_session(self) -> str:
        """Force start a new session (for page reload/restart)"""
        self.current_session_id = self.create_new_session()
        print(f"Started new session: {self.current_session_id}")
        return self.current_session_id

# Global history manager instance
history_manager = PostgreSQLHistoryManager()



