/* ==========================================================================
   Interactive Priority Zones Map — Leaflet
   Shows 6 European cities with cooling potential on hover
   v3: Light basemap, white tooltips, updated palette.
   ========================================================================== */

(function () {
  'use strict';

  var container = document.getElementById('priority-map-container');
  if (!container) return;

  // Replace static image with map div
  var img = container.querySelector('img');
  if (img) img.style.display = 'none';

  var mapDiv = document.createElement('div');
  mapDiv.id = 'priority-map';
  mapDiv.style.width = '100%';
  mapDiv.style.height = '65vh';
  mapDiv.style.borderRadius = '6px';
  mapDiv.style.overflow = 'hidden';
  mapDiv.style.border = '1px solid rgba(0,0,0,0.08)';
  mapDiv.style.boxShadow = '0 4px 12px rgba(0,0,0,0.08)';
  container.appendChild(mapDiv);

  // City data
  var cities = [
    {
      name: 'Amsterdam',
      lat: 52.3689, lon: 4.8889,
      climate: 'Cfb – Oceanic',
      cooling: '-0.92\u00B0C',
      priority: '38.5%',
      keyDriver: 'Building density',
      topRisk: '0.6% severe',
      color: '#3a6fa0'
    },
    {
      name: 'Athens',
      lat: 37.9875, lon: 23.7195,
      climate: 'Csa – Mediterranean',
      cooling: '-1.45\u00B0C',
      priority: '78.2%',
      keyDriver: 'Topography + sealed surfaces',
      topRisk: '1.6% severe',
      color: '#c4513a'
    },
    {
      name: 'Barcelona',
      lat: 41.3851, lon: 2.1863,
      climate: 'Csa – Mediterranean',
      cooling: '-1.31\u00B0C',
      priority: '42.1%',
      keyDriver: 'Coastal-hill wind channels',
      topRisk: '0.2% very high',
      color: '#c4513a'
    },
    {
      name: 'Berlin',
      lat: 52.5162, lon: 13.3923,
      climate: 'Cfb/Dfb – Transitional',
      cooling: '-1.08\u00B0C',
      priority: '33.7%',
      keyDriver: 'Impervious fraction',
      topRisk: '0.3% very high',
      color: '#555555'
    },
    {
      name: 'Madrid',
      lat: 40.4219, lon: -3.7244,
      climate: 'Csa/BSk – Semi-arid trans.',
      cooling: '-0.89\u00B0C',
      priority: '20.9%',
      keyDriver: 'Semi-arid drought stress',
      topRisk: '0.7% severe',
      color: '#555555'
    },
    {
      name: 'Paris',
      lat: 48.8575, lon: 2.3621,
      climate: 'Cfb – Oceanic',
      cooling: '-1.22\u00B0C',
      priority: '45.3%',
      keyDriver: 'Centre-periphery heat gradient',
      topRisk: '0.4% very high',
      color: '#3a6fa0'
    }
  ];

  // Initialize map
  var map = L.map('priority-map', {
    center: [46.5, 10],
    zoom: 4,
    zoomControl: false,
    attributionControl: false,
    scrollWheelZoom: false,
    dragging: true
  });

  // Light tile layer
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19
  }).addTo(map);

  // Add city markers
  cities.forEach(function (city) {
    var coolingNum = parseFloat(city.cooling);
    var radius = Math.abs(coolingNum) * 20 + 8;

    // Circle marker
    var marker = L.circleMarker([city.lat, city.lon], {
      radius: radius,
      fillColor: city.color,
      color: '#ffffff',
      weight: 2,
      opacity: 0.9,
      fillOpacity: 0.6
    }).addTo(map);

    // Tooltip (on hover) — light theme
    marker.bindTooltip(
      '<div style="font-family:Inter,sans-serif;font-size:14px;line-height:1.5;min-width:200px;color:#1a1a1a">' +
      '<strong style="font-size:16px;color:' + city.color + '">' + city.name + '</strong>' +
      '<span style="float:right;font-size:12px;color:#888">' + city.climate + '</span><br>' +
      '<span style="color:#c4513a;font-size:20px;font-weight:700">' + city.cooling + '</span>' +
      ' <span style="color:#555">predicted cooling</span><br>' +
      '<span style="color:#2a7f6f">' + city.priority + '</span>' +
      ' <span style="color:#888">priority zone coverage</span><br>' +
      '<span style="color:#888;font-size:12px">Key driver: ' + city.keyDriver + '</span>' +
      '</div>',
      {
        direction: 'top',
        offset: [0, -radius],
        className: 'city-tooltip',
        permanent: false,
        opacity: 0.95
      }
    );

    // City label
    L.marker([city.lat, city.lon], {
      icon: L.divIcon({
        className: 'city-label',
        html: '<span>' + city.name + '</span>',
        iconSize: [100, 20],
        iconAnchor: [50, -radius - 4]
      })
    }).addTo(map);
  });

  // Re-initialize map when slide becomes active (reveal.js lazy rendering)
  if (typeof Reveal !== 'undefined') {
    Reveal.on('slidechanged', function () {
      setTimeout(function () { map.invalidateSize(); }, 300);
    });
  }
})();
