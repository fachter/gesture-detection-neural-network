from sanic import Sanic
from sanic.response import html
from gesturenetwork.app_engine.app_engine import AppEngine


# ========== Slideshow config ==============
from gesturenetwork.app_engine.app_utils import get_camera_index_from_argument

app = Sanic("slideshow_server")
app.static("/static", "./static")


def handle_optional_gestures(gesture: str):
    print("Gesture:", gesture)
    if gesture == "rotate" or gesture == "rotate_right":
        return "rotate_right"
    elif gesture == "rotate_left":
        return "rotate_left"
    elif gesture == "swipe_right":
        return "left"
    elif gesture == "swipe_left":
        return "right"
    elif gesture == "flip_table":
        return "down"
    elif gesture == "spread":
        return "zoom_in"
    elif gesture == "pinch":
        return "zoom_out"


engine = AppEngine("../network_configs/network_config_optional_app.pkl",
                   action_handler_function=handle_optional_gestures,
                   old=True,
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
