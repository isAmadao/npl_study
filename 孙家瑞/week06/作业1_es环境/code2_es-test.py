import json

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ApiError


# ===================== 1. 连接 Elasticsearch =====================
def connect_es():
    """连接 ES 并验证连接"""
    try:
        # 本地部署：host 为 localhost，端口 9200
        es = Elasticsearch(
            ["http://127.0.0.1:9200"],
            # 若设置了用户名密码，需添加：
            basic_auth=("elastic", "changeme"),
            verify_certs=False,  # 8.x 必须关闭 SSL 验证（本地 ES 无证书）
            ssl_show_warn=False,  # 关闭 SSL 警告
        )
        es = es.options(request_timeout=2)
        # 验证连接
        if es.ping():
            print("✅ ES 连接成功！")
            # 打印 ES 版本信息
            info = es.info()
            print(f"ES 版本：{info['version']['number']}")
            return es
        else:
            print("❌ ES 连接失败！")
            return None
    except ApiError as e:
        print(f"❌ ES 连接异常：{e}")
        return None


# ===================== 2. 索引管理（创建/删除/查看） =====================
def test_index_operations(es):
    """测试索引的创建、查看、删除"""
    index_name = "test_books"  # 测试索引名
    print("\n===== 测试索引操作 =====")

    # 1. 创建索引（指定映射）
    index_mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text", "analyzer": "ik_max_word"},  # 文本类型，支持分词
                "author": {"type": "keyword"},  # 关键字类型，不分词
                "price": {"type": "float"},  # 浮点型
                "publish_date": {"type": "date", "format": "yyyy-MM-dd"},  # 日期型
                "category": {"type": "keyword"}  # 分类（关键字）
            }
        },
        "settings": {
            "number_of_shards": 1,  # 分片数（单机设1）
            "number_of_replicas": 0  # 副本数（单机设0）
        }
    }

    # 先删除已存在的索引（避免冲突）
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"已删除原有索引：{index_name}")

    # 创建新索引
    try:
        es.indices.create(index=index_name, body=index_mapping)
        print(f"✅ 成功创建索引：{index_name}")
    except ApiError as e:
        print(f"❌ 创建索引失败：{e}")

    # 查看索引信息
    try:
        index_info = es.indices.get(index=index_name)
        print(f"✅ 索引 {index_name} 信息：\n{json.dumps(index_info[index_name]['mappings'], indent=2)}")
    except ApiError as e:
        print(f"❌ 查看索引失败：{e}")

    return index_name


# ===================== 3. 文档 CRUD（增删改查） =====================
def test_document_operations(es, index_name):
    """测试文档的新增、查询、更新、删除"""
    print("\n===== 测试文档操作 =====")

    # 1. 新增文档（指定ID）
    doc1_id = "1"
    doc1 = {
        "title": "Python编程：从入门到实践",
        "author": "埃里克·马瑟斯",
        "price": 89.0,
        "publish_date": "2020-01-01",
        "category": "编程"
    }
    try:
        es.index(index=index_name, id=doc1_id, body=doc1)
        print(f"✅ 新增文档成功（ID={doc1_id}）")
    except ApiError as e:
        print(f"❌ 新增文档失败：{e}")

    # 批量新增文档
    docs = [
        {"index": {"_id": "2"}},
        {"title": "Java核心技术", "author": "Cay Horstmann", "price": 129.0, "publish_date": "2021-05-01",
         "category": "编程"},
        {"index": {"_id": "3"}},
        {"title": "数据分析实战", "author": "Peter Bruce", "price": 79.0, "publish_date": "2019-08-01",
         "category": "数据分析"},
        {"index": {"_id": "4"}},
        {"title": "Python数据分析", "author": "Wes McKinney", "price": 99.0, "publish_date": "2022-03-01",
         "category": "数据分析"}
    ]
    try:
        es.bulk(body=docs, index=index_name)
        print("✅ 批量新增文档成功")
    except ApiError as e:
        print(f"❌ 批量新增文档失败：{e}")

    # 2. 查询文档（按ID）
    try:
        doc = es.get(index=index_name, id=doc1_id)
        print(f"✅ 查询文档（ID={doc1_id}）：\n{json.dumps(doc['_source'], indent=2, ensure_ascii=False)}")
    except ApiError as e:
        print(f"❌ 查询文档失败：{e}")

    # 3. 更新文档（部分字段）
    try:
        es.update(
            index=index_name,
            id=doc1_id,
            body={"doc": {"price": 79.9, "publish_date": "2020-02-01"}}
        )
        # 重新查询验证更新
        updated_doc = es.get(index=index_name, id=doc1_id)
        print(
            f"✅ 更新文档成功（ID={doc1_id}），更新后内容：\n{json.dumps(updated_doc['_source'], indent=2, ensure_ascii=False)}")
    except ApiError as e:
        print(f"❌ 更新文档失败：{e}")

    # 4. 删除文档
    try:
        es.delete(index=index_name, id=doc1_id)
        # 验证删除（查询不到即为成功）
        if not es.exists(index=index_name, id=doc1_id):
            print(f"✅ 删除文档成功（ID={doc1_id}）")
    except ApiError as e:
        print(f"❌ 删除文档失败：{e}")


# ===================== 4. 测试搜索功能（全文检索/过滤/排序） =====================
def test_search(es, index_name):
    """测试 ES 核心搜索功能"""
    print("\n===== 测试搜索功能 =====")

    # 1. 全文检索：搜索标题包含「Python」的文档
    search_body1 = {
        "query": {
            "match": {
                "title": "Python"
            }
        },
        "sort": [{"price": "asc"}]  # 按价格升序
    }
    try:
        result1 = es.search(index=index_name, body=search_body1)
        print(f"✅ 全文检索（标题含Python）：共找到 {result1['hits']['total']['value']} 条结果")
        for hit in result1['hits']['hits']:
            print(f"  - 文档ID：{hit['_id']}，标题：{hit['_source']['title']}，价格：{hit['_source']['price']}")
    except ApiError as e:
        print(f"❌ 全文检索失败：{e}")

    # 2. 过滤查询：分类为「数据分析」且价格 < 90 的文档
    search_body2 = {
        "query": {
            "bool": {
                "must": [{"match": {"category": "数据分析"}}],
                "filter": [{"range": {"price": {"lt": 90}}}]
            }
        }
    }
    try:
        result2 = es.search(index=index_name, body=search_body2)
        print(f"\n✅ 过滤查询（数据分析+价格<90）：共找到 {result2['hits']['total']['value']} 条结果")
        for hit in result2['hits']['hits']:
            print(f"  - 文档ID：{hit['_id']}，标题：{hit['_source']['title']}，价格：{hit['_source']['price']}")
    except ApiError as e:
        print(f"❌ 过滤查询失败：{e}")


# ===================== 5. 测试聚合功能 =====================
def test_aggregation(es, index_name):
    """测试 ES 聚合功能（统计/分组）"""
    print("\n===== 测试聚合功能 =====")

    # 聚合查询：按分类分组，计算每组的平均价格、最高价格
    agg_body = {
        "size": 0,  # 不返回原始文档，只返回聚合结果
        "aggs": {
            "category_group": {
                "terms": {"field": "category", "size": 10},  # 按分类分组
                "aggs": {
                    "avg_price": {"avg": {"field": "price"}},  # 平均价格
                    "max_price": {"max": {"field": "price"}}  # 最高价格
                }
            }
        }
    }
    try:
        result = es.search(index=index_name, body=agg_body)
        print("✅ 聚合结果（按分类统计价格）：")
        for bucket in result['aggregations']['category_group']['buckets']:
            category = bucket['key']
            avg_price = bucket['avg_price']['value']
            max_price = bucket['max_price']['value']
            print(f"  - 分类：{category}，文档数：{bucket['doc_count']}，平均价格：{avg_price:.2f}，最高价格：{max_price:.2f}")
    except ApiError as e:
        print(f"❌ 聚合查询失败：{e}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 1. 连接 ES
    es_client = connect_es()
    if not es_client:
        exit(1)

    # 2. 测试索引操作
    index_name = test_index_operations(es_client)

    # 3. 测试文档 CRUD
    test_document_operations(es_client, index_name)

    # 4. 测试搜索功能
    test_search(es_client, index_name)

    # 5. 测试聚合功能
    test_aggregation(es_client, index_name)

    # 6. 清理测试数据（可选）
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        print(f"\n✅ 已删除测试索引：{index_name}")

    print("\n🎉 所有 ES 核心功能测试完成！")
