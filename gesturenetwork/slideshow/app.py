from sanic import Sanic
from sanic.response import html
from gesturenetwork.app_engine.app_engine import AppEngine

# ========== Slideshow config ==============

# slideshow_root_path = os.path.dirname(__file__) + "/slideshow/"
from gesturenetwork.app_engine.app_utils import get_camera_index_from_argument

app = Sanic("slideshow_server")
app.static("/static", "./static")


def handle_mandatory_gestures(gesture: str):
    print("Gesture:", gesture)
    if gesture == "rotate":
        return "rotate"
    elif gesture == "swipe_right":
        return "left"
    elif gesture == "swipe_left":
        return "right"


engine = AppEngine("../network_configs/network_config_mandatory_app.pkl",
                   action_handler_function=handle_mandatory_gestures,
                   camera_index=get_camera_index_from_argument())


@app.route("/")
async def index(_request):
    return html(open("slideshow.html", "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    await engine.emit_events(ws)


if __name__ == "__main__":
    app.add_task(engine.run())
    app.run(host="0.0.0.0", debug=True)
