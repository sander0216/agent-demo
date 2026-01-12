import chromadb
from pypdf import PdfReader
from openai import OpenAI
import os
import json  # 引入 json 库，用来处理存档

# =================配置区域=================
API_KEY = "sk-a821aba999ce40cc9c8349cf7ac2871d"  # 换成你的 Key
PDF_PATH = "data.pdf"
DB_PATH = "./my_vector_db"  # 向量数据库存储路径
HISTORY_FILE = "chat_history.json"  # 聊天记录存档文件
# =========================================

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# --- 1. 持久化向量数据库 (修改点) ---
# 使用 PersistentClient，数据会保存在文件夹里
print("正在连接知识库...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="my_knowledge")

# 只有当数据库里没东西时，才去读 PDF (避免重复处理)
if collection.count() == 0:
    print("首次运行，正在读取 PDF 并构建索引...")
    try:
        reader = PdfReader(PDF_PATH)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()

        chunks = [pdf_text[i:i + 300] for i in range(0, len(pdf_text), 300)]
        collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
        print("索引构建完成！")
    except:
        print("未找到 PDF，跳过索引构建。")
else:
    print("检测到已有知识库，直接加载！(无需重新读 PDF)")

# --- 2. 加载历史记录 (读档) ---
if os.path.exists(HISTORY_FILE):
    print("发现旧的聊天记录，正在加载...")
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        history_messages = json.load(f)
else:
    print("没有发现旧记录，开始新对话。")
    history_messages = [
        {"role": "system", "content": "你是一个助手。请结合上下文和参考资料回答问题。"}
    ]

print("=" * 30)
print("开始对话 (输入 'q' 退出，退出时会自动保存)")

# --- 3. 对话循环 ---
try:
    while True:
        question = input("\n用户: ")
        if question.lower() == 'q':
            break

        # RAG 检索
        results = collection.query(query_texts=[question], n_results=2)
        if results['documents']:
            retrieved_text = " ".join(results['documents'][0])
        else:
            retrieved_text = "无相关资料"

        full_user_input = f"【资料】{retrieved_text}\n【问题】{question}"

        history_messages.append({"role": "user", "content": full_user_input})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=history_messages,
            stream=False
        )

        answer = response.choices[0].message.content
        print(f"AI: {answer}")

        history_messages.append({"role": "assistant", "content": answer})

        # --- 实时保存 (可选：每聊一句存一次，防止程序崩溃) ---
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_messages, f, ensure_ascii=False, indent=2)

except KeyboardInterrupt:
    # 比如你按了 Ctrl+C 强制退出
    print("\n程序强制中断...")

print("对话结束，记录已保存。")