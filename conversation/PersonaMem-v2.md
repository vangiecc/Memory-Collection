---
title: PersonaMem-v2
date: 2026-10-04
---
回答的问题：
用户只把chatbot当作完成，并不会显式表达自己的偏好，而偏好有可能隐藏在具体的任务之中（撰写电子邮件、翻译文本、查找信息）
大型语言模型能多好地理解用户，尤其是从漫长的对话记录中解读出用户的隐性用户画像和偏好，从而提供个性化回复？

理解并记住用户偏好->更好地理解用户意图->提供更个性化的回复

pipeline

1.persona构造与偏好生成

原始数据集：personahub
https://huggingface.co/datasets/proj-persona/PersonaHub
随机抽取一条persona

```text
A software developer who is looking for a way to simplify the integration of GPRS technology into their embedded system designs. They are interested in developing a stable and efficient software stack for an embedded system and are willing to invest time and effort into finding a solution that meets their requirements. They are looking for a product that is easy to use and has minimal requirements for technical knowledge, while also being able to provide accurate and reliable data transmission. They are also interested in finding a product that is compatible with other network protocols and can be easily integrated into existing systems.
```


用 LLM 逐步扩展：人口学字段（性别/种族/性取向按预设权重分布采样注入）、然后一次生成 30 条刻板+ 30 条反刻板 + 30 条中性偏好（共90条偏好），再逐条用 LLM 校验清洗冲突；

另外追加 therapy 背景、病史、仿真敏感信息（伪 SSN/银行卡/API key）。         

每条偏好用 LLM 归类到开放 topic（全局计数器统计分布）。

2.隐式对话生成     

每条偏好变成一段场景对话（邮件润色/翻译/倾诉/知识提问等 9 类），prompt 反复强调"偏好必须隐式表达、需要推理才能读出"。                  

几个精心设计的测试维度：

• 刻板 vs 反刻板：生成前用 self-verify（让模型根据偏好反猜 persona，与真实 persona 比对）剔除"伪反刻板"；    

• who=others（33%）：偏好属于"我朋友"，测模型会不会错记到用户头上；

• 偏好更新（67%）：生成一段新对话把偏好反转（updated: True, prev_pref: 旧值），测记忆更新；                        

• ask_to_forget（33%）：追加"请忘掉这个"请求，测记忆删除；

• 敏感信息：模拟用户误贴 .env/表单，测隐私泄漏。             

 3. QA 生成与质量控制（qa_generator.py）              

• 每段对话出一道四选一题：正确答案个性化，干扰项含一个故意按人口学刻板印象回答的选项。          

• validate_qa 五重校验
	其中最重要的是无上下文测试：不给历史裸问模型，如果模型不看历史也能答对，说明这题没有区分度，直接丢弃。
	其余校验问题泄题、答案对齐、干扰项污染、格式清洁。 

4. 长上下文构建（contexts_builder.py
   
• 排序保证因果性：旧偏好对话在前、更新对话在后、"请忘记"块在最后。                                                                       
 • 32k：超预算时智能裁剪，含评测问题的块和被更新引用的块受保护。                                                                          
 • 128k：不裁原内容，改用 GSM8K/Omni-MATH/BigCodeBench 生成的无关对话（解题、debug）在对话块边界插入填充，做"大海捞针"式抗干扰测试。      

• system message 里放人口学画像 JSON（注意：不含偏好列表），多模态版内嵌 base64 图片。                                                   
    
6. 图片匹配（image_matcher.py）

离线：每张 photobook 图片让 VLM 推断"拍摄者画像"（推断什么样的人最有可能拍下这张图片）并 embedding；
	输出包含：年龄范围、国籍/族裔、社会经济地位、职业、地理位置

在线：与 persona embedding 余弦相似 top-k （k取随机数3-8）作为该用户的"相册"，再让 VLM 读图反推偏好、生成"发照片提问"的多模态对话。 


