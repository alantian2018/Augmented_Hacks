// Creating map options
var mapOptions = {
   center: [37.3230, -122.0322],
   zoom: 5
}

// Creating a map object
var map = new L.map('map', mapOptions);
var popmap = new L.map('pop-map', mapOptions);

// Creating a Layer object
var layer = new L.TileLayer('https://api.maptiler.com/maps/hybrid/{z}/{x}/{y}.jpg?key=hM01WZAOFjwbYPSEZYOj', {
    attribution: '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>'
});

var noGreen = `
	void main(void) {
		// Classic texel look-up (fetch the texture "pixel" color for this fragment)
		vec4 texelColour = texture2D(uTexture0, vec2(vTextureCoords.s, vTextureCoords.t));

		// If uncommented, this would output the image "as is"
		// gl_FragColor = texelColour;
        float maxC = 0.7;
		// Let's mix the colours a little bit, inverting the red and green channels.
		if(texelColour.r > maxC && texelColour.g > maxC && texelColour.b > maxC) {
		    gl_FragColor = vec4(texelColour.r, texelColour.g, texelColour.b, 0.0);
		}else {
		    gl_FragColor = vec4(texelColour.r, texelColour.g, texelColour.b, 1.0);
        }
	}
`

var poplayer = L.tileLayer.gl({
   tileUrls: ['https://tile.casa.ucl.ac.uk/duncan/WorldPopDen2015b/{z}/{x}/{y}.png'],
   fragmentShader: noGreen,
   attribution: '<a href="https://luminocity3d.org/WorldPopDen/#3/14.77/13.27">luminocity3d.org</a>'
});
// 'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png');

// Adding layer to the map

// var opacityOptions = {
//    opacityBaseControl: {
//       options: {
//         sliderImageUrl: "https://unpkg.com/leaflet-transparency@0.0.2/images/opacity-slider3d7.png",
//         backgroundColor: "rgba(0, 0, 0, 0.9)",
//         position: 'topright',
//       },
//    }
// }

// var baseOpacity = new L.Control.OpacitySlider(null, opacityOptions.opacityBaseControl.options);
// baseOpacity.addTo(popmap);

// d3.json(Globals.resourceWithPath("countries.geo.json"), function (json){
// 	function style(feature) {
// 		return {
// 			fillColor: "#E3E3E3",
// 			weight: 1,
// 			opacity: 0.4,
// 			color: 'white',
// 			fillOpacity: 0.3
// 		};
// 	}
// 	C.geojson = L.geoJson(json, {
// 		onEachFeature: onEachFeature,
// 		style : style
// 	}).addTo(popmap);
// });

var geoJson = new L.GeoJSON.AJAX("smalltest.json");
console.log(geoJson);
geoJson.addTo(popmap);

map.addLayer(layer); 
popmap.addLayer(poplayer);
map.sync(popmap);
popmap.sync(map);

var slider = document.getElementById("myRange");
var output = document.getElementById("demo");
output.innerHTML = slider.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider.oninput = function() {
  output.innerHTML = this.value;
}