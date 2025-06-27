"""
@Author:         FishNoNoNo
@Version:        1.0.1
@Date:           2025-06-27
"""

from openai import OpenAI
import time
from flask import Flask, Response, jsonify, render_template, request
import json
import random

# 读取配置

config={}

with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


app = Flask(__name__)

client = OpenAI(
    api_key=config['model']['api_key'],
    base_url=config['model']['base_url'],
)

book_content = config["book_content"]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_question", methods=["GET"])
def get_question():
    # 从查询参数获取消息
    user_message = request.args.get("message", "请根据教材内容提出一个问题")

    random_num=random.randint(0,len(book_content))

    content=f"你是一个老师,你需要依据{book_content[random_num]}向我提问,我会进行回答,然后你需要对我的回答进行评分."

    system_message = {"role": "system", "content": content}

    def event_generator():
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                system_message,
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )

        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                content = content.replace("\n", "").replace("*", "")
                # 正确构建 SSE 格式: data: <content>\n\n
                yield f"data: {content}\n\n"
            time.sleep(0.1)

    # 设置正确的响应类型
    return Response(
        event_generator(),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/evaluate_answer", methods=["POST"])
def evaluate_answer():
    # 获取JSON数据
    data = request.get_json()
    question = data.get("question", "")
    answer = data.get("answer", "")

    if not question or not answer:
        return jsonify({"error": "缺少问题或答案"}), 400

    # 构造评估提示
    evaluation_prompt = f"""
    请根据教材内容评估以下回答的质量：
    
    问题：{question}
    学生回答：{answer}
    
    请从以下几个方面进行评估：
    1. 准确性(0-10分)
    2. 完整性(0-10分)
    3. 表述清晰度(0-10分)
    4. 总体评分(0-10分)
    5. 反馈建议(指出错误和不足，提出改进建议)
    
    最后以JSON格式返回评估结果,包含以下字段:
    - accuracy_score
    - completeness_score
    - clarity_score
    - overall_score
    - feedback
    """

    # 调用API进行评估
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个政治学教授，负责评估学生对政治理论问题的回答",
                },
                {"role": "user", "content": evaluation_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        # 解析JSON响应
        result = response.choices[0].message.content
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
