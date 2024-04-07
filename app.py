from flask import Flask, render_template, request
import sys
from common import model
from chatbot import Chatbot
from chorong_system_role import system_role, instruction
from function_calling import func_specs, FunctionCalling  # 단일 함수 호출

# from parallel_function_calling import FunctionCalling, tools  # 병렬적 함수 호출

# chorongGak 인스턴스 생성
chorongGak = Chatbot(
    model=model.advanced, system_role=system_role, instruction=instruction  # advanced
)

application = Flask(__name__)

func_calling = FunctionCalling(model=model.advanced)


@application.route("/")
def hello():
    return render_template("welcome.html")


@application.route("/welcome")
def welcome():
    return "Hello ChorongBot!"


@application.route("/chat-app")
def chat_app():
    return render_template("chat.html")


@application.route("/chat-api", methods=["POST"])
def chat_api():
    request_message = request.json["request_message"]
    print("request_message:", request_message)
    chorongGak.add_user_message(request_message)

    # 챗GPT에게 함수사양을 토대로 사용자 메시지에 호응하는 함수 정보를 분석해달라고 요청
    analyzed_dict = func_calling.analyze(request_message, func_specs)  # 단일 함수 호출
    # analyzed, analyzed_dict = func_calling.analyze(request_message, tools) # 병렬적 함수 호춝
    # 챗GPT가 함수 호출이 필요하다고 분석했는지 여부 체크
    if analyzed_dict.get("function_call"):  # 단일 함수 호출
        # if analyzed_dict.get("tool_calls"): # 병렬적 함수 호출
        # 챗GPT가 분석해준 대로 함수 호출
        response = func_calling.run(
            analyzed_dict, chorongGak.context[:]
        )  # 단일 함수 호출
        # response = func_calling.run(analyzed, analyzed_dict, chorongGak.context[:]) # 병렬적 함수 호출
        chorongGak.add_response(response)
    else:
        response = chorongGak.send_request()
        chorongGak.add_response(response)

    response_message = chorongGak.get_response_content()
    chorongGak.handle_token_limit(response)
    chorongGak.clean_context()
    print("response_message:", response_message)
    return {"response_message": response_message}


if __name__ == "__main__":
    application.config["TEMPLATES_AUTO_RELOAD"] = True
    application.jinja_env.auto_reload = True
    application.run(host="0.0.0.0", port=int(sys.argv[1]))
