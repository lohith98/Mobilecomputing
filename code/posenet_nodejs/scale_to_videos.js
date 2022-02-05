const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');

const {
    createCanvas, Image
} = require('canvas');

let fs = require('fs');
let path = require('path');
let output_dir = path.join(__dirname, '..', '..', 'demo', 'posenets')

const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
});

// Function Params to decide Pose Estimation
const imageScaleFactor = 0.5;
const outputStride = 16;
const flipHorizontal = false;
const defaultQuantBytes = 4;

const defaultMobileNetMultiplier = 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 801;

// Posenet instance params used
let guiState = {
    net: null,
    model: {
        architecture: 'MobileNetV1',
        outputStride: defaultMobileNetStride,
        inputResolution: defaultMobileNetInputResolution,
        multiplier: defaultMobileNetMultiplier,
        quantBytes: defaultQuantBytes,
    },
    image: 'tennis_in_crowd.jpg',
    multiPoseDetection: {
        minPartConfidence: 0.1,
        minPoseConfidence: 0.2,
        nmsRadius: 20.0,
        maxDetections: 15,
    },
    showKeypoints: true,
    showSkeleton: true,
    showBoundingBox: false,
};

// Path to the location where frames of all the video files are stored
getArgumentValue = process.argv[2];
if (getArgumentValue === undefined) {
    photo_path_to_frames = 'C:/Users/Ram/Desktop/asl_fingerspelling/demo/tmp'
} else {
    photo_path_to_frames = getArgumentValue;
}

console.log(photo_path_to_frames);


/**
 * Asynchronous Function to decide poses for a set of images and storing it in a single json file
 *
 * @returns {Promise<void>}
 * @param photo_path
 * @param choice
 */
function cascading_images_pose_estimation(photo_path) {
    console.log("Starting to estimate pose values for list of Video frames in the given path : " + photo_path);
    let config;
    let choice = "1"
    if (choice === "1") {
        console.log("Loading model : MobileNetV1");
        config = {
            architecture: guiState.model.architecture,
            outputStride: guiState.model.outputStride,
            inputResolution: guiState.model.inputResolution,
            multiplier: guiState.model.multiplier,
            quantBytes: guiState.model.quantBytes
        };
    } else {
        console.log("Loading Model : ResNet50");
        config = {
            architecture: "ResNet50",
            outputStride: guiState.model.outputStride,
            inputResolution: guiState.model.inputResolution,
            quantBytes: guiState.model.quantBytes
        };
    }

    const single_net = posenet.load(config);

    let canvas;
    let input;
    let ctx;
    console.log('Reading directory - ', photo_path)
    fs.readdir(photo_path, function (err, items) {
        let func_path = "";
        for (let i = 0; i < items.length; i++) {
            console.log("Processing dir - " + items[i])
            filename = items[i].split(".")[0]
            let pose_list = [];
            if (path.extname(items[i]) === "") {
                func_path = photo_path + "/" + items[i] + "/";
                let images = fs.readdirSync(func_path)
                let length = images.length;
                console.log("Total frames for " + items[i] + " are " + length)
                for (let j = 0; j < length - 1; j++) {
                    //console.log("Reading image - " + func_path + images[j])
                    let image = loadImage(func_path + images[j]);
                    // image.src = func_path + i + ".png";
                    //console.log("Image read")
                    canvas = createCanvas(image.width, image.height);
                    ctx = canvas.getContext('2d');
                    console.log("image - " + image)
                    ctx.drawImage(image, 0, 0);
                    input = tf.browser.fromPixels(canvas);
                    let pose = single_net.estimateSinglePose(input, imageScaleFactor, flipHorizontal, outputStride);
                    pose_list.push(pose);
                }

                //console.log("Final Pose list - " + pose_list)
                fs.writeFileSync(func_path + "key_points.json", JSON.stringify(pose_list));
                console.log("Key Points File for \"" + items[i] + "\" has been created in " + func_path + "key_points.json");
            }
        }
    });

    console.log("Done.")
}

async function loadImage(path) {
    let image = new Image();
    const promise = new Promise((resolve, reject) => {
        image.onload = () => {
            resolve(image);
        };
    });
    image.src = path;
    return promise;
}


cascading_images_pose_estimation(photo_path_to_frames)
readline.close();
process.stdin.destroy();