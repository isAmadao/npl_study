#!/usr/bin/env python3
"""
本地多代理系统演示：主代理 + 情感分类代理 + 实体识别代理
使用模拟数据避免外部API依赖
"""

import asyncio
import random
from datetime import datetime

class LocalAgent:
    """本地代理基类"""
    
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
    
    async def process(self, input_text):
        """处理输入文本"""
        # 模拟处理延迟
        await asyncio.sleep(0.5)
        return f"{self.name}处理结果: {input_text}"

class SentimentAgent(LocalAgent):
    """情感分类代理"""
    
    def __init__(self):
        super().__init__(
            "情感分类代理",
            "专门分析文本的情感倾向（积极、消极、中性）"
        )
        
        # 情感关键词库
        self.positive_words = ["好", "棒", "优秀", "满意", "开心", "愉快", "喜欢", "爱"]
        self.negative_words = ["差", "糟糕", "失望", "难过", "生气", "讨厌", "恨", "烦"]
    
    async def process(self, input_text):
        """分析文本情感"""
        await asyncio.sleep(0.3)
        
        # 简单情感分析逻辑
        positive_count = sum(1 for word in self.positive_words if word in input_text)
        negative_count = sum(1 for word in self.negative_words if word in input_text)
        
        if positive_count > negative_count:
            sentiment = "积极"
            intensity = "强烈" if positive_count > 3 else "中等" if positive_count > 1 else "轻微"
        elif negative_count > positive_count:
            sentiment = "消极"
            intensity = "强烈" if negative_count > 3 else "中等" if negative_count > 1 else "轻微"
        else:
            sentiment = "中性"
            intensity = "中性"
        
        # 识别情感关键词
        keywords = []
        for word in self.positive_words + self.negative_words:
            if word in input_text:
                keywords.append(word)
        
        result = f"""
情感分析报告：
- 情感倾向: {sentiment}
- 情感强度: {intensity}
- 识别关键词: {', '.join(keywords) if keywords else '无明确情感关键词'}
- 分析文本: "{input_text}"

详细解释:
文本整体表现出{sentiment}情感倾向，强度为{intensity}。
"""
        
        return result

class EntityAgent(LocalAgent):
    """实体识别代理"""
    
    def __init__(self):
        super().__init__(
            "实体识别代理", 
            "专门识别文本中的人名、地名、组织名等实体"
        )
        
        # 实体识别规则
        self.entity_patterns = {
            "人名": ["张三", "李四", "王五", "赵六", "小明", "小红"],
            "地名": ["北京", "上海", "广州", "深圳", "杭州", "成都"],
            "组织名": ["腾讯", "阿里巴巴", "百度", "华为", "苹果", "谷歌"],
            "产品名": ["iPhone", "微信", "淘宝", "支付宝", "Windows"]
        }
    
    async def process(self, input_text):
        """识别文本中的实体"""
        await asyncio.sleep(0.4)
        
        entities = []
        
        # 识别各类实体
        for entity_type, patterns in self.entity_patterns.items():
            found_entities = []
            for pattern in patterns:
                if pattern in input_text:
                    found_entities.append(pattern)
            
            if found_entities:
                entities.append({
                    "type": entity_type,
                    "names": found_entities,
                    "count": len(found_entities)
                })
        
        result = f"""
实体识别报告：
- 分析文本: "{input_text}"
- 识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

识别到的实体:"""
        
        if not entities:
            result += "\n- 未识别到明确的实体"
        else:
            for entity in entities:
                result += f"\n- {entity['type']}: {', '.join(entity['names'])} (共{entity['count']}个)"
        
        result += f"""

统计信息:
- 总实体数: {sum(e['count'] for e in entities)}
- 实体类型数: {len(entities)}
"""
        
        return result

class MasterAgent(LocalAgent):
    """主代理 - 负责路由"""
    
    def __init__(self, sentiment_agent, entity_agent):
        super().__init__(
            "主代理",
            "负责接收用户请求并路由到合适的子代理"
        )
        self.sentiment_agent = sentiment_agent
        self.entity_agent = entity_agent
    
    async def route_request(self, input_text):
        """路由用户请求"""
        await asyncio.sleep(0.2)
        
        # 简单的路由逻辑
        sentiment_triggers = ["情感", "情绪", "感受", "心情", "态度", "喜欢", "讨厌"]
        entity_triggers = ["人名", "地名", "公司", "组织", "产品", "品牌", "实体"]
        
        has_sentiment = any(trigger in input_text for trigger in sentiment_triggers)
        has_entity = any(trigger in input_text for trigger in entity_triggers)
        
        if has_sentiment and not has_entity:
            selected_agent = self.sentiment_agent
            reason = "检测到情感相关关键词"
        elif has_entity and not has_sentiment:
            selected_agent = self.entity_agent
            reason = "检测到实体识别相关关键词"
        elif has_sentiment and has_entity:
            # 两者都有，根据主要意图选择
            if len([t for t in sentiment_triggers if t in input_text]) > len([t for t in entity_triggers if t in input_text]):
                selected_agent = self.sentiment_agent
                reason = "情感相关关键词更多"
            else:
                selected_agent = self.entity_agent
                reason = "实体相关关键词更多"
        else:
            # 无法识别，由主代理处理
            return f"主代理直接响应: 我收到了您的消息『{input_text}』，但无法确定具体处理需求。请明确说明您需要情感分析还是实体识别。"
        
        # 执行子代理任务
        agent_result = await selected_agent.process(input_text)
        
        return f"""路由决策:
- 选择的代理: {selected_agent.name}
- 选择理由: {reason}

{agent_result}"""

class MultiAgentSystem:
    """多代理系统"""
    
    def __init__(self):
        # 创建子代理
        self.sentiment_agent = SentimentAgent()
        self.entity_agent = EntityAgent()
        
        # 创建主代理
        self.master_agent = MasterAgent(self.sentiment_agent, self.entity_agent)
    
    async def process_request(self, user_input):
        """处理用户请求"""
        return await self.master_agent.route_request(user_input)
    
    async def interactive_chat(self):
        """交互式聊天界面"""
        print("=== 本地多代理系统演示 ===")
        print("系统包含：主代理 + 情感分类代理 + 实体识别代理")
        print("输入 '退出' 或 'quit' 结束对话")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("您：")
                
                if user_input.lower() in ['退出', 'quit', 'exit']:
                    print("系统：再见！")
                    break
                
                if not user_input.strip():
                    continue
                
                print("系统：处理中...")
                
                # 处理用户请求
                result = await self.process_request(user_input)
                
                print(f"系统：{result}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n系统：对话结束")
                break
            except Exception as e:
                print(f"系统：处理时出现错误 - {e}")

async def demo():
    """演示函数"""
    
    # 创建多代理系统
    system = MultiAgentSystem()
    
    # 测试用例
    test_cases = [
        "我觉得今天的天气真好，心情特别愉快",
        "北京是中国的首都，有很多名胜古迹",
        "苹果公司发布了新款iPhone，但价格太贵了",
        "帮我分析一下这段话的情感：我对这次会议的结果感到非常失望",
        "张三和李四去了上海参加会议"
    ]
    
    print("=== 多代理系统测试演示 ===")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"输入: {test_input}")
        
        result = await system.process_request(test_input)
        print(f"输出: {result}")
        print("-" * 50)
        
        # 添加延迟以便观察
        await asyncio.sleep(1)
    
    # 启动交互式聊天
    print("\n=== 开始交互式对话演示 ===")
    await system.interactive_chat()

if __name__ == "__main__":
    print("启动本地多代理系统...")
    asyncio.run(demo())