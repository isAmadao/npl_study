import sqlite3

"""
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,       -- 'user' / 'assistant' / 'system'
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    tokens INTEGER            -- 可选，记录 token 用量
);
CREATE INDEX idx_session ON conversations(session_id);
"""
"""
用户输入 → 加载历史（从DB:load_history） → 追加新用户消息(save_message) → 裁剪至上下文限制 → 调用LLM API → 保存助手回复 → 返回结果
"""
def save_message(session_id, role, content):

    conn = sqlite3.connect("chat_history.db")
    conn.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def load_history(session_id, max_messages=20):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp LIMIT ?",
        (session_id, max_messages)
    )
    history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return history