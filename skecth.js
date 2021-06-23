var model1;
var model2;

var loadFile = function(event) {
var image = document.getElementById('input_img');
image.src = URL.createObjectURL(event.target.files[0]);
};

async function Predict() {
model1 = await tf.loadGraphModel('hand_seg_seperable_conv_256_face_to_face_aug_web/model.json'); 
model2 = await tf.loadGraphModel('face_to_face_aug_4st_epoch_web/model.json');
zero = tf.zeros([1,256,256,3])
model1.predict(zero)
model2.predict(zero)
console.log("Done_with_zero")
console.log("Model_loded")
let image = document.getElementById("input_img") 
let out_canv1 = document.getElementById("out_put_canvas1")
var start1 = window.performance.now(); 
let tensorImg1 =   tf.browser.fromPixels(image).resizeNearestNeighbor([256, 256]).div(tf.scalar(255)).expandDims();
let prediction1 = await model1.predict(tensorImg1);
let mask1 = await prediction1.reshape([256,256]);
var end1 = window.performance.now();

let out_canv2 = document.getElementById("out_put_canvas2")
var start2 = window.performance.now(); 
let tensorImg2 =   tf.browser.fromPixels(image).resizeNearestNeighbor([256, 256]).div(tf.scalar(255)).expandDims();
let prediction2 = await model2.predict(tensorImg2);
let mask2 = await prediction2.reshape([256,256]);
var end2 = window.performance.now();

var extime1 = end1-start1;
var output_message_b = ""
var output_message_e = " ms"
var output_message1 = output_message_b.concat(extime1,output_message_e);
extime1 = extime1.toString();
tf.browser.toPixels(mask1,out_canv1);
console.log(`Execution time1: ${end1 - start1} ms`);
console.log("Done_with_prediction");
document.getElementById("demo1").innerHTML = output_message1;

var end2 = window.performance.now();
var extime2 = end2-start2;
var output_message2 = output_message_b.concat(extime2,output_message_e);
extime2 = extime2.toString();
tf.browser.toPixels(mask2,out_canv2);
console.log(`Execution time2: ${end2 - start2} ms`);
console.log("Done_with_prediction");
document.getElementById("demo2").innerHTML = output_message2;
};