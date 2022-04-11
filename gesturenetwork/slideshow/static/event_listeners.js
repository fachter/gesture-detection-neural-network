let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function (event) {
    let currentSlide
    console.log("received '" + event.data + "' event")
    switch (event.data) {
        case "right":
            Reveal.right();
            break;
        case "left":
            Reveal.left();
            break;
        case "down":
            Reveal.down()
            break;
        case "rotate":
        case "rotate_right":
            currentSlide = Reveal.getCurrentSlide();
            rotateRotatables(currentSlide, true);
            break;
        case "rotate_left":
            currentSlide = Reveal.getCurrentSlide();
            rotateRotatables(currentSlide, false);
            break;
        case "zoom_in":
            // increases zoom by 10%
            zoom(10); // `zoom()` is defined in helper_methods.js
            break;
        case "zoom_out":
            // decreases zoom by 10%
            zoom(-10); // `zoom()` is defined in helper_methods.js
            break;
        default:
            console.debug(`unknown message received from server: ${event.data}`);
    }
};
