from sanic import Sanic
from sanic.response import html
from gesturenetwork.app_engine.app_engine import AppEngine
from gesturenetwork.app_engine.app_utils import get_camera_index_from_argument

app = Sanic("tetris_server")
app.static("/static", "./static")


def handle_tetris_gestures(gesture: str):
    print("Gesture:", gesture)
    if gesture == "swipe_left":
        return "left"
    elif gesture == "swipe_right":
        return "right"
    elif gesture == "rotate":
        return gesture
    elif gesture == "flip_table":
        return "up"


engine = AppEngine("../network_configs/network_config_optional_app.pkl",
                   action_handler_function=handle_tetris_gestures, old=True,
                   camera_index=get_camera_index_from_argument())


@app.route("/")
async def index(_request):
    return html(open("tetris.html", "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    await engine.emit_events(ws)


if __name__ == "__main__":
    app.add_task(engine.run())
    app.run(host="localhost", debug=True, port=8001)
