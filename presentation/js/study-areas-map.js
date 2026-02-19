/* ==========================================================================
   Study Areas Location Map — Leaflet
   Shows 6 European cities colour-coded by climate zone.
   Simpler than the priority-zones map: location + climate, no results data.
   Initialises lazily when the slide appears; fades out on fragment click.
   ========================================================================== */

(function () {
  'use strict';

  var container = document.getElementById('study-areas-map-container');
  if (!container) return;

  var mapDiv = document.createElement('div');
  mapDiv.id = 'study-areas-map';
  mapDiv.style.cssText =
    'width:100%;height:65vh;border-radius:6px;overflow:hidden;' +
    'border:1px solid rgba(0,0,0,0.08);box-shadow:0 4px 12px rgba(0,0,0,0.08);';
  container.appendChild(mapDiv);

  var cities = [
    { name: 'Amsterdam', lat: 52.3689, lon: 4.8889,  climate: 'Cfb – Oceanic',              color: '#3a6fa0' },
    { name: 'Berlin',    lat: 52.5162, lon: 13.3923,  climate: 'Cfb/Dfb – Transitional',     color: '#555555' },
    { name: 'Paris',     lat: 48.8575, lon: 2.3621,   climate: 'Cfb – Oceanic',              color: '#3a6fa0' },
    { name: 'Athens',    lat: 37.9875, lon: 23.7195,  climate: 'Csa – Mediterranean',        color: '#c4513a' },
    { name: 'Barcelona', lat: 41.3851, lon: 2.1863,   climate: 'Csa – Mediterranean',        color: '#c4513a' },
    { name: 'Madrid',    lat: 40.4219, lon: -3.7244,  climate: 'Csa/BSk – Semi-arid trans.', color: '#555555' }
  ];

  var map = null;

  function initMap() {
    if (map) return;

    map = L.map('study-areas-map', {
      center: [46.0, 10.0],
      zoom: 4,
      zoomControl: false,
      attributionControl: false,
      scrollWheelZoom: false,
      dragging: false,
      doubleClickZoom: false,
      boxZoom: false,
      keyboard: false
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19
    }).addTo(map);

    cities.forEach(function (city) {
      // Circle marker — uniform size, climate-zone colour
      var marker = L.circleMarker([city.lat, city.lon], {
        radius: 14,
        fillColor: city.color,
        color: '#ffffff',
        weight: 2.5,
        opacity: 1,
        fillOpacity: 0.7
      }).addTo(map);

      // Tooltip — city name + climate zone
      marker.bindTooltip(
        '<div style="font-family:Inter,sans-serif;font-size:14px;line-height:1.4;min-width:140px;color:#1a1a1a">' +
        '<strong style="font-size:15px;color:' + city.color + '">' + city.name + '</strong><br>' +
        '<span style="color:#888;font-size:12px">' + city.climate + '</span>' +
        '</div>',
        {
          direction: 'top',
          offset: [0, -16],
          className: 'city-tooltip',
          permanent: false,
          opacity: 0.95
        }
      );

      // Permanent city label above marker
      L.marker([city.lat, city.lon], {
        icon: L.divIcon({
          className: 'city-label',
          html: '<span>' + city.name + '</span>',
          iconSize: [100, 20],
          iconAnchor: [50, -18]
        })
      }).addTo(map);
    });
  }

  // Lazy-init: only create the map when the slide is visible
  if (typeof Reveal !== 'undefined') {
    Reveal.on('slidechanged', function (event) {
      if (event.currentSlide.querySelector('#study-areas-map-container')) {
        setTimeout(function () {
          initMap();
          if (map) map.invalidateSize();
        }, 300);
      }
    });

    // Also handle direct navigation
    Reveal.on('ready', function () {
      var cur = Reveal.getCurrentSlide();
      if (cur && cur.querySelector('#study-areas-map-container')) {
        setTimeout(function () {
          initMap();
          if (map) map.invalidateSize();
        }, 300);
      }
    });
  }
})();
