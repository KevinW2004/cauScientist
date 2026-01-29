import wikipediaapi

# 必须设置 User-Agent，否则会被维基百科封禁
wiki = wikipediaapi.Wikipedia(
    user_agent="MyGradProject/1.0 (contact: your_email@example.com)", language="en"
)

page = wiki.page("Tuberculosis")  # 直接填词条名
if page.exists():
    print(page.summary)  # 拿到纯净的摘要
    print(page.text)  # 拿到纯净的全文
    # 然后把 page.text 存进你的 RAG
