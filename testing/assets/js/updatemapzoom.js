function updateZoom(map, mapLayer, popLayer)
{
    if (map.getAttribute(zoom) > 10) {
        map.addLayer(mapLayer);
    }
    else
        map.addLayer(popLayer);
}