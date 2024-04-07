from flask import Flask, render_template, request
import sys
from common import model
from chatbot import Chatbot
from chorong_system_role import system_role, instruction

# chorongGak 인스턴스 생성
chorongGak = Chatbot(
    model=model.advanced, system_role=system_role, instruction=instruction  # advanced
)

application = Flask(__name__)


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
    response = chorongGak.send_request()
    chorongGak.add_response(response)
    response_message = chorongGak.get_response_content()
    chorongGak.handle_token_limit(response)
    chorongGak.clean_context()
    print("response_message:", response_message)
    return {"response_message": response_message}


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=int(sys.argv[1]))
