const uid = function (i) {
    return function () {
        return "generated_id-" + (++i);
    };
}(0);

const rotateRotatables = function (rotationAngles) {
    return function (slide, rotate_right = true) {
        const rotatables = Array.from(slide.getElementsByClassName("rotatable"))
        if (rotatables.length > 0) {
            rotatables.forEach(function (elem) {
                if (!elem.id) elem.id = uid();

                if (!rotationAngles[elem.id]) {
                    rotationAngles[elem.id] = 0
                }

                if (rotate_right){
                    let ankle = (90);

                    new_rotation = rotationAngles[elem.id] + ankle
                    elem.style.transform = "rotate(" + (new_rotation) + "deg)"
                    rotationAngles[elem.id] = new_rotation
                } else {
                    let ankle = -(90);

                    new_rotation = rotationAngles[elem.id] + ankle
                    elem.style.transform = "rotate(" + (new_rotation) + "deg)"
                    rotationAngles[elem.id] = new_rotation
                }

            });
        }
    }
}({})

const zoom = function (zoomStepSize) {
    body = document.getElementsByTagName("body")[0];
    const currentZoom = Number(body.style.zoom.replace("%", "")) || 100;
    newZoom = Math.max(currentZoom + zoomStepSize, 40); // don't go lower than 40% zoom
    body.style.zoom = newZoom + "%"
}
