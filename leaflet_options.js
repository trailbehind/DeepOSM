var style = {
        "clickable": true,
        "color": "#00D",
        "fillColor": "#00D",
        "weight": 2.0,
        "opacity": 0.3,
        "fillOpacity": 0.2,
    };
    var hoverStyle = {
        "fillOpacity": 0.5
    };

    var map = L.map('mainMap').setView([37.81385, -119.48559], 12);
    var geojsonURL = 'http://vector.mapzen.com/osm/roads/{z}/{x}/{y}.json?api_key=vector-tiles-NsMiwBc';
    var geojsonTileLayer = new L.TileLayer.GeoJSON(geojsonURL, {
            clipTiles: true,
            unique: function (feature) {
                return feature.id; 
            }
        }, {
            style: style,
            onEachFeature: function (feature, layer) {
            }
        }
    );
    map.addLayer(geojsonTileLayer);
