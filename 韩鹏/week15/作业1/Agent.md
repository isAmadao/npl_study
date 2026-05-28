# 作业要求
使用claude code结合05-multimodal-rag-chatbot的需求，进行vibe coding一份，需要实现所有接口；
写清楚需求
写测试逻辑
cc逐步完成，架构和初步代码完成即可

# 开发过程
1. 开发一个图文知识库项目，要求如下：
    - 使用Python开发，本地python环境使用conda管理，使用名称为py310的环境进行开发和测试
2. 项目文件结构要求如下：
    - uploads文件夹：存储上传文件
    - SQLAlchemy ORM文件：基于 SQLite 数据库，定义一个用于管理上传文件处理状态的表，名称为file，数据库文件名为db.db，存放在项目根目录下
    - 首页文件：使用streamlit编写，包含两个菜单：文件管理、图文对话
    - 文件管理模块页面：使用streamlit编写，实现以下功能：
      - 上传pdf/docx文档功能，自动将文档保存到uploads文件夹，并在sqllite中的file表记录文件基本信息，再通知kafka进行文件处理。
      - 页面要显示已上传的所有文档，并支持删除功能
    - 图文对话模块文件：使用streamlit编写，实现以下功能：
      - 以对话框形式与大模型进行交互，后端根据输入内容，自动从Milvus数据库中提取相关信息并传给大模型，并返回最终信息
      - 大模型连接信息：model="qwen3.5-plus",api_key="",base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    - 上传文件异步处理文件：负责消费kafka中待处理文件的任务，将文件进行切分、向量化、存储到向量库中
      - 文本使用bge-small-zh-v1.5模型向量化,图文使用jina-clip-v2模型向量化。模型库已经下载，存储在项目models文件夹下
      - kafka本地服务地址：localhost:9092
      - 向量数据库使用线上Milvus库，信息为：uri="", token=""，collection_name="rag_data_new"
3. 根据以上要求进行开发，执行步骤要求如下
   - 业务分析，形成闭环
   - 系统架构设计审核
   - 生成多步骤开发计划，并形成md格式的文档，保存在项目根目录下，文件名为plan.md
   - 根据开发计划md文件，按照步骤进行开发，注释完整
   - 代码单元测试
   - 代码审核，只对重大问题，如逻辑错误等才进行优化