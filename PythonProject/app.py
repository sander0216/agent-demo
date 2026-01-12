import streamlit as st
import chromadb
from pypdf import PdfReader
from openai import OpenAI
import os
import json

# =================配置区域=================
# ⚠️ 注意：实际部署时尽量不要把 Key 直接写在代码里，建议用环境变量
API_KEY = "sk-a821aba999ce40cc9c8349cf7ac2871d"
PDF_PATH = "data.pdf"
DB_PATH = "./my_vector_db"
HISTORY_FILE = "chat_history.json"
# =========================================

# 设置网页标题
st.set_page_config(page_title="我的 RAG 助手", layout="centered")
st.title("🤖 智能 RAG 问答助手")


# --- 1. 初始化资源 (使用缓存，避免每次操作都重读 PDF) ---
@st.cache_resource
def get_vector_db():
    print("正在连接知识库...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name="my_knowledge")

    # 只有当数据库为空时，才读取 PDF
    if collection.count() == 0:
        if os.path.exists(PDF_PATH):
            print("首次运行，正在读取 PDF 并构建索引...")
            reader = PdfReader(PDF_PATH)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()

            # 切分文本
            chunks = [pdf_text[i:i + 300] for i in range(0, len(pdf_text), 300)]
            collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
            print("索引构建完成！")
        else:
            print("警告：未找到 PDF 文件。")
    return collection


# 加载数据库 (这一步只会在第一次运行时比较慢)
collection = get_vector_db()
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# --- 2. 管理聊天记录 (Session State) ---
# 初始化 session_state，如果不存在则创建
if "messages" not in st.session_state:
    # 尝试从本地加载旧记录
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = [
            {"role": "system", "content": "你是一个助手。请结合上下文和参考资料回答问题。"}
        ]

# --- 3. 渲染聊天界面 ---
# 遍历历史记录并在网页上显示 (跳过 system 消息)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        # 如果是 user 发的消息，但包含了【资料】前缀，我们在界面上只显示【问题】部分，比较美观
        content_to_show = msg["content"]
        if msg["role"] == "user" and "【资料】" in content_to_show:
            # 简单的分割逻辑，只显示“【问题】”后面的内容
            try:
                content_to_show = content_to_show.split("【问题】")[1]
            except:
                pass  # 如果分割失败，就显示原文

        with st.chat_message(msg["role"]):
            st.markdown(content_to_show)

# --- 4. 处理用户输入 ---
# st.chat_input 相当于原来的 input()
if user_query := st.chat_input("请输入你的问题..."):

    # 1. 在界面上立即显示用户的问题
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. RAG 检索逻辑
    results = collection.query(query_texts=[user_query], n_results=2)
    if results['documents'] and results['documents'][0]:
        retrieved_text = " ".join(results['documents'][0])
    else:
        retrieved_text = "无相关资料"

    # 3. 构造发给 AI 的完整 Prompt
    full_prompt = f"【资料】{retrieved_text}\n【问题】{user_query}"

    # 将完整 Prompt 加入历史记录 (为了让 AI 记住上下文)
    st.session_state.messages.append({"role": "user", "content": full_prompt})

    # 4. 调用 AI 并流式输出结果
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 为了体验更好，这里把 stream 改成了 True (可选)
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=st.session_state.messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")  # 加个光标特效

        message_placeholder.markdown(full_response)

    # 5. 保存 AI 回复
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 6. 持久化保存到 JSON (每聊一次存一次)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)