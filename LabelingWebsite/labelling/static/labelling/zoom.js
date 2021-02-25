/*
# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

# core code for marking
*/

var canvas = document.getElementsByTagName('canvas')[0];
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
canvas.style.display = "none";

var overlay = document.getElementById('overlay');
overlay.style.display = "";

var gkhead = new Image;
var points = [];
var removedPoints = []
var ctx = canvas.getContext('2d');
var curFillStyle = "black";
var lastFillStyle = "black";

var buttons = document.getElementsByTagName('button');

if(curMenu!="GBM")
{
	curFillStyle = "green";
	lastFillStyle = "green";
}

if(labels && labels.length >0)
{

	points = labels
}
for (var i = 0; i < buttons.length; i++) {
	var button = buttons[i];
	if (i == 0) {
		button.innerHTML = '<font color="red"> <b><i>' + button.innerText + '<i><b> </font>';
	}
	else {
		button.innerHTML = '<font color="white">' + button.innerText + '</font>'
	}

}


function PostData(){
    var data = {'pid':patient_id, 'curMenu':curMenu, 'points': JSON.stringify(points)};
	$.post(post_url,
		data, function(response)
		{
			alert(response);},
	"json");

    alert("Your request is processed!");
}

function OnColor(btn, style) {
	if (style != "undo") {
		var buttons = document.getElementsByTagName('button');
		for (var i = 0; i < buttons.length; i++) {
			var button = buttons[i];
			button.innerHTML = '<font color="white">' + button.innerText + '</font>'
			console.log(button.value)
		}
		btn.innerHTML = '<font color="red"> <b><i>' + btn.innerText + '<i><b> </font>';
	}
	else {
		if (curFillStyle == "eraser") {
			if (removedPoints.length > 0) {
				points.push(removedPoints.pop());
				redraw();
				return;
			}
		}
		if (points.length > 0) {
			points.splice(-1, 1);
			redraw();
			return;
		}


	}



	curFillStyle = style;
	if (style != "eraser") {
		removedPoints = []
	}

}


function redraw() {

	if (!gkhead.complete) return;
	// Clear the entire canvas
	var p1 = ctx.transformedPoint(0, 0);
	var p2 = ctx.transformedPoint(canvas.width, canvas.height);
	ctx.clearRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);

	ctx.save();
	ctx.setTransform(1, 0, 0, 1, 0, 0);
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.restore();

	ctx.drawImage(gkhead, 0, 0);

	ctx.save();
	for (var i = 0; i < points.length; i++) {
		ctx.fillStyle = points[i].style;
		ctx.beginPath();
		ctx.arc(points[i].x, points[i].y, Math.round(marker_size/2), 0, 2 * Math.PI);
		ctx.fill();
	}
	ctx.restore();



}


gkhead.onload = function () {

	canvas.style.display = "";
	overlay.style.display = "none";
	trackTransforms(ctx);
	redraw();

	var lastX = canvas.width / 2, lastY = canvas.height / 2;

	var dragStart, dragged;

	canvas.addEventListener('mousedown', function (evt) {
		document.body.style.mozUserSelect = document.body.style.webkitUserSelect = document.body.style.userSelect = 'none';
		lastX = evt.offsetX || (evt.pageX - canvas.offsetLeft);
		lastY = evt.offsetY || (evt.pageY - canvas.offsetTop);
		dragStart = ctx.transformedPoint(lastX, lastY);
		dragged = false;
	}, false);

	canvas.addEventListener('mousemove', function (evt) {
		lastX = evt.offsetX || (evt.pageX - canvas.offsetLeft);
		lastY = evt.offsetY || (evt.pageY - canvas.offsetTop);
		dragged = true;
		if (dragStart) {
			var pt = ctx.transformedPoint(lastX, lastY);
			ctx.translate(pt.x - dragStart.x, pt.y - dragStart.y);
			// console.log(lastX,lastY, pt.x - dragStart.x, pt.y - dragStart.y);
			redraw();
		}
	}, false);

	canvas.addEventListener('mouseup', function (evt) {
		dragStart = null;
		if (!dragged) zoom(evt.shiftKey ? -1 : evt.button);
	}, false);

	var scaleFactor = 1.1;

	var zoom = function (clicks) {
		var pt = ctx.transformedPoint(lastX, lastY);
		ctx.translate(pt.x, pt.y);
		var factor = Math.pow(scaleFactor, clicks);
		// console.log(pt.x, pt.y);
		if(clicks == 1 || clicks==2)
		{
			return;
		}
		else if (clicks == 0) 
		{
			if (curFillStyle == "eraser") {
				for (var i = 0; i < points.length; i++) {

					if (Math.abs(points[i].x - pt.x) < marker_size/2 && Math.abs(points[i].y - pt.y) < marker_size/2) {
						removedPoints.push(points[i]);
						// console.log('removed', points[i].x, points[i].y);
						points.splice(i, 1);
						break;
					}


				}
			}
			else {

				if(pt.x < 0 || pt.y < 0 || pt.x > gkhead.naturalWidth || pt.y > gkhead.naturalHeight)
				{
					console.log("invalid click point!");
				}
				else
				{

					points.push({ x: Math.round(pt.x), y: Math.round(pt.y), style: curFillStyle });
				}
				
			}

		}
		else {
			ctx.scale(factor, factor);
		}

		ctx.translate(-pt.x, -pt.y);



		redraw();
	}

	var handleScroll = function (evt) {
		var delta = evt.wheelDelta ? evt.wheelDelta / 40 : evt.detail ? -evt.detail : 0;
		if (delta) zoom(delta);
		return evt.preventDefault() && false;
	};

	canvas.addEventListener('DOMMouseScroll', handleScroll, false);
	canvas.addEventListener('mousewheel', handleScroll, false);

	for(var i=0; i< 10; i++){
		zoom(-3);
		ctx.translate(-1*(gkhead.naturalWidth / 20), -1*(gkhead.naturalHeight / 20));
		// console.log(-1*(gkhead.naturalWidth / 20), -1*(gkhead.naturalHeight / 20));
		redraw();
	}
	
	
};

gkhead.src =  label_image;

// Adds ctx.getTransform() - returns an SVGMatrix
// Adds ctx.transformedPoint(x,y) - returns an SVGPoint
function trackTransforms(ctx) {
	var svg = document.createElementNS("http://www.w3.org/2000/svg", 'svg');
	var xform = svg.createSVGMatrix();
	ctx.getTransform = function () { return xform; };

	var savedTransforms = [];
	var save = ctx.save;
	ctx.save = function () {
		savedTransforms.push(xform.translate(0, 0));
		return save.call(ctx);
	};

	var restore = ctx.restore;
	ctx.restore = function () {
		xform = savedTransforms.pop();
		return restore.call(ctx);
	};

	var scale = ctx.scale;
	ctx.scale = function (sx, sy) {
		xform = xform.scaleNonUniform(sx, sy);
		return scale.call(ctx, sx, sy);
	};

	var rotate = ctx.rotate;
	ctx.rotate = function (radians) {
		xform = xform.rotate(radians * 180 / Math.PI);
		return rotate.call(ctx, radians);
	};

	var translate = ctx.translate;
	ctx.translate = function (dx, dy) {
		xform = xform.translate(dx, dy);
		return translate.call(ctx, dx, dy);
	};

	var transform = ctx.transform;
	ctx.transform = function (a, b, c, d, e, f) {
		var m2 = svg.createSVGMatrix();
		m2.a = a; m2.b = b; m2.c = c; m2.d = d; m2.e = e; m2.f = f;
		xform = xform.multiply(m2);
		return transform.call(ctx, a, b, c, d, e, f);
	};

	var setTransform = ctx.setTransform;
	ctx.setTransform = function (a, b, c, d, e, f) {
		xform.a = a;
		xform.b = b;
		xform.c = c;
		xform.d = d;
		xform.e = e;
		xform.f = f;
		return setTransform.call(ctx, a, b, c, d, e, f);
	};

	var pt = svg.createSVGPoint();
	ctx.transformedPoint = function (x, y) {
		pt.x = x; pt.y = y;
		return pt.matrixTransform(xform.inverse());
	}
}
