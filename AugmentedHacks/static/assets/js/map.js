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
		vec4 texelColour = texture2D(uTexture0, vec2(vTextureCoords.s, vTextureCoords.t));

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
});

var legend = L.control({position: 'bottomleft'});
function getColor(d) {
    return d > 40000  ? '#ff9e00' :
           d > 22000  ? '#ff0000' :
           d > 15000  ? '#990049' :
           d >10000   ? '#00309f' :
           d > 1000   ? '#a8e3e5' :
           d > 0   ?    '#f4fbf2' :
                        'white';
}
legend.onAdd = function (map) {

    var div = L.DomUtil.create('div', 'info legend'),
        grades = [0, 1000, 10000, 15000, 22000,40000],
        grades2=["0","1k","10k","15k","22k","40k"]
        labels = [];
    div.innerHTML+='<h style="font-size:18px;">Pop. Density</h><br><br>'
    // loop through our density intervals and generate a label with a colored square for each interval
    for (var i = 0; i < grades.length; i++) {
        div.innerHTML +=
            '<i style="background:' + getColor(grades[i] + 1) + '"></i> ' +
            grades2[i] + (grades2[i + 1] ? '&ndash;' + grades2[i + 1] + '<br>' : '+');
    }

    return div;
};


var geoJson = new L.GeoJSON.AJAX(static_url + "assets/countries.geo.json");
// console.log(geoJson);
geoJson.addTo(popmap);

map.addLayer(layer);
popmap.addLayer(poplayer);
map.sync(popmap);
popmap.sync(map);
legend.addTo(map);
var slider = document.getElementById("myRange");
var output = document.getElementById("demo");
output.innerHTML = slider.value;
slider.oninput = function () {
    output.innerHTML = this.value;
}