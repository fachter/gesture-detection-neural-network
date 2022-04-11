let socket = new WebSocket("ws://localhost:8001/events");

socket.onmessage = function (event) {
    switch (event.data) {
        case "right":
            console.log("received 'right' event");
            tetris.right();
            break;
        case "left":
            console.log("received 'left' event");
            tetris.left();
            break;
        case "rotate":
            console.log("received 'rotate' event");
            tetris.up();
            break;
        case "up":
            console.log("received 'up' event")
            tetris.space()
            break;
        case "zoom_in":
            console.log("received 'zoom_in' event");
            break;
        case "zoom_out":
            console.log("received 'zoom_out' event");
            break;
        default:
            console.debug(`unknown message received from server: ${event.data}`);
    }
};
