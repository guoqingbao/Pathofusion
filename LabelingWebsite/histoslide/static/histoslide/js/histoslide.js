// histoslide functions
var viewer;
var viewer_image;
var viewer_is_new;
var bm_center;
var bm_zoom;
var bm_image;
var bm_goto;
var slidevendor;
var annotations;

function get_slideid(slidelink) {
    var filename = slidelink.split('/').pop();
    filename = filename.replace(".dzi","");
    return filename
}

function open_slide(link) {
    // Enable waiting for function to finish
    var r = $.Deferred();
    // Load info objects
    var image;
    image=link;
    // Create viewer if necessary
    if (!viewer) {
        viewer = new OpenSeadragon({
            id: "slidepane",
            prefixUrl: "/static/histoslide/images/",
            showNavigator: true,
            animationTime: 0.5,
            blendTime: 0.1,
            constrainDuringPan: false,
            maxZoomPixelRatio: 2,
            minPixelRatio: 0.5,
            minZoomLevel: 1,
            visibilityRatio: 1,
            zoomPerScroll: 2
        });
        viewer.addHandler("open", function() {
            viewer.source.minLevel = 8;
            /* Start zoomed in, then return to home position after
               loading.  Workaround for blurry viewport on initial
               load (OpenSeadragon #95). */
            var center = new OpenSeadragon.Point(0.5,
                    1 / (2 * viewer.source.aspectRatio));
            viewer.viewport.zoomTo(2, center, true);
            viewer_is_new = true;
            /* Ensure we receive update-viewport events, OpenSeadragon
               #94 */
            viewer.drawer.viewer = viewer;
        });
        viewer.addHandler("update-viewport", function() {
            if (viewer_is_new) {
                setTimeout(function() {
                    if (viewer.viewport) {
                        viewer.viewport.goHome(false);
                    }
                }, 5);
                viewer_is_new = false;
            }
        });
        viewer.addHandler("home", function() {
            if (bm_goto) {
                setTimeout(function() {
                    if (viewer.viewport) {
                        viewer.viewport.zoomTo(bm_zoom,bm_center, false);
                    }
                }, 200);
                bm_goto = false;
            }
        });
        viewer.scalebar({
            type: OpenSeadragon.ScalebarType.MICROSCOPY,
            pixelsPerMeter: 1000000,
            minWidth: "160px",
            location: OpenSeadragon.ScalebarLocation.BOTTOM_LEFT,
            xOffset: 5,
            yOffset: 10,
            stayInsideImage: true,
            color: "rgb(150, 150, 150)",
            fontColor: "rgb(100, 100, 100)",
            backgroundColor: "rgb(255, 255, 255)",
            fontSize: "small",
            barThickness: 2
        });
    }

    // Load slide
    if (image!==viewer_image) {
        $.getJSON(image + ".json", {}, function(slideprop) {
            slidevendor = slideprop.vendor;
            if (!slideprop.mppx || slideprop.mppx === 0.0) {
                viewer.scalebar({
                    pixelsPerMeter:0.0
                });
            } else {
                viewer.scalebar({
                    pixelsPerMeter:1000000.0/slideprop.mppx
                });
            };

        });

        viewer.open(image);
        viewer_image=image;
    } else {
        if (bm_goto) {
            viewer.viewport.goHome(bm_zoom,false);
        }
    }
}

$(".slide_link").click(function(ev) {
    slideid=$(this).attr("id").substring(5);
    open_slide("/histoslide/"+slideid+".dzi");
});



// CSS doesn't provide a good way to specify a div of height
// (100% - height(header))
$(window).resize(function() {
    $('#content').height($(window).height() -
                $('#header').outerHeight() - 20);
}).resize();

