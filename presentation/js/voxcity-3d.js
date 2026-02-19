/**
 * VoxCity 6-City 3D Viewer — Three.js
 *
 * Renders all 6 European cities as 3D voxel models in a 2x3 grid.
 * Two modes: Geometry (architectural) and Solar Irradiance (ground-only heatmap).
 * Uses InstancedMesh per city for performance.
 *
 * v6: Multi-city layout, solar on ground only, architectural palette.
 */

(function () {
  'use strict';

  var scene, camera, renderer, controls;
  var cityMeshes = {};
  var cityData = {};
  var currentMode = 'geometry';
  var isInitialized = false;
  var animationId = null;

  var CITIES = [
    { name: 'amsterdam', label: 'Amsterdam', col: 0, row: 0 },
    { name: 'athens',    label: 'Athens',    col: 1, row: 0 },
    { name: 'barcelona', label: 'Barcelona', col: 2, row: 0 },
    { name: 'berlin',    label: 'Berlin',    col: 0, row: 1 },
    { name: 'madrid',    label: 'Madrid',    col: 1, row: 1 },
    { name: 'paris',     label: 'Paris',     col: 2, row: 1 }
  ];

  // Grid layout: 3 columns × 2 rows, spacing between cities
  var GRID_SPACING = 220;
  var GRID_OFFSET_X = -GRID_SPACING;  // center the 3-column grid
  var GRID_OFFSET_Z = -GRID_SPACING * 0.5;

  // ── Geometry mode — architectural model palette ──
  var CLASS_COLORS = {
    0: [0.72, 0.71, 0.69],   // ground
    1: [0.78, 0.77, 0.75],   // wall
    2: [0.35, 0.42, 0.28],   // tree — olive green
    3: [0.65, 0.64, 0.62],   // road
    4: [0.82, 0.81, 0.79]    // building — light grey
  };

  // ── Solar mode: structural colors for non-ground ──
  var SOLAR_STRUCTURAL = {
    1: [0.62, 0.61, 0.59],
    2: [0.32, 0.38, 0.26],
    4: [0.68, 0.67, 0.65]
  };

  // ── Inferno colormap for solar irradiance ──
  var INFERNO = [
    [0.001, 0.000, 0.014],
    [0.106, 0.024, 0.277],
    [0.280, 0.040, 0.430],
    [0.479, 0.090, 0.397],
    [0.660, 0.180, 0.310],
    [0.830, 0.310, 0.200],
    [0.950, 0.500, 0.100],
    [0.990, 0.700, 0.100],
    [0.990, 0.880, 0.350],
    [0.988, 0.998, 0.645]
  ];

  function solarColor(t) {
    t = Math.max(0, Math.min(1, t));
    var idx = t * (INFERNO.length - 1);
    var lo = Math.floor(idx);
    var hi = Math.min(lo + 1, INFERNO.length - 1);
    var f = idx - lo;
    return [
      INFERNO[lo][0] + (INFERNO[hi][0] - INFERNO[lo][0]) * f,
      INFERNO[lo][1] + (INFERNO[hi][1] - INFERNO[lo][1]) * f,
      INFERNO[lo][2] + (INFERNO[hi][2] - INFERNO[lo][2]) * f
    ];
  }

  function getVoxelColor(v, mode, hasSolar) {
    var cls = v[3];
    if (mode === 'solar' && hasSolar && v.length > 4) {
      if (cls === 0 || cls === 3) return solarColor(v[4] / 255);
      return SOLAR_STRUCTURAL[cls] || SOLAR_STRUCTURAL[4];
    }
    return CLASS_COLORS[cls] || CLASS_COLORS[0];
  }

  function checkWebGL() {
    try {
      var c = document.createElement('canvas');
      return !!(c.getContext('webgl') || c.getContext('experimental-webgl'));
    } catch (e) { return false; }
  }

  function showFallback() {
    var container = document.getElementById('voxcity-container');
    var fallback = document.getElementById('voxcity-fallback');
    var canvas = document.getElementById('voxcity-canvas');
    if (fallback) fallback.style.display = 'block';
    if (canvas) canvas.style.display = 'none';
    if (container) {
      var c = container.querySelector('.voxcity-controls');
      var h = container.querySelector('.voxcity-hint');
      if (c) c.style.display = 'none';
      if (h) h.style.display = 'none';
    }
  }

  function initScene() {
    if (typeof THREE === 'undefined') { showFallback(); return false; }
    if (!checkWebGL()) { showFallback(); return false; }

    var container = document.getElementById('voxcity-container');
    var canvas = document.getElementById('voxcity-canvas');
    if (!container || !canvas) return false;

    var width = container.clientWidth || 1200;
    var height = container.clientHeight || 600;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xedecea);
    scene.fog = new THREE.FogExp2(0xedecea, 0.0008);

    // Elevated perspective to see all 6 cities
    camera = new THREE.PerspectiveCamera(40, width / height, 1, 5000);
    camera.position.set(350, 400, 500);

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    if (THREE.OrbitControls) {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.25;
      controls.maxPolarAngle = Math.PI * 0.55;
      controls.minPolarAngle = Math.PI * 0.1;
      controls.minDistance = 200;
      controls.maxDistance = 1200;
      controls.target.set(GRID_SPACING * 0.5, 0, 0);
    }

    // Lighting — soft studio
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    scene.add(new THREE.HemisphereLight(0xf0f0f5, 0xd0cfc8, 0.35));

    var key = new THREE.DirectionalLight(0xfff5e8, 0.55);
    key.position.set(200, 400, 200);
    scene.add(key);

    var fill = new THREE.DirectionalLight(0xe8f0ff, 0.2);
    fill.position.set(-200, 200, -200);
    scene.add(fill);

    return true;
  }

  function buildCityMesh(data, cityInfo) {
    var voxels = data.voxels;
    var dims = data.dims;
    var count = voxels.length;

    var geometry = new THREE.BoxGeometry(1, 1, 1);
    var material = new THREE.MeshPhongMaterial({
      vertexColors: false,
      shininess: 6,
      specular: new THREE.Color(0x1a1a1a),
      flatShading: true
    });

    var mesh = new THREE.InstancedMesh(geometry, material, count);
    var dummy = new THREE.Object3D();
    var color = new THREE.Color();

    // Center each city's voxels around (0,0) then offset to grid position
    var cx = dims[0] / 2;
    var cz = dims[1] / 2;
    var cy = dims[2] / 2;

    // Scale to normalize city sizes (so they fit uniformly in the grid)
    var maxDim = Math.max(dims[0], dims[1]);
    var scale = 160 / maxDim;  // normalize to ~160 units wide

    var gridX = GRID_OFFSET_X + cityInfo.col * GRID_SPACING;
    var gridZ = GRID_OFFSET_Z + cityInfo.row * GRID_SPACING;

    for (var i = 0; i < count; i++) {
      var v = voxels[i];
      var x = (v[0] - cx) * scale + gridX;
      var y = (v[2]) * scale;
      var z = (v[1] - cz) * scale + gridZ;

      dummy.position.set(x, y, z);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      var rgb = getVoxelColor(v, currentMode, data.hasSolar);
      color.setRGB(rgb[0], rgb[1], rgb[2]);
      mesh.setColorAt(i, color);
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    scene.add(mesh);

    // City label using sprite
    var labelCanvas = document.createElement('canvas');
    labelCanvas.width = 512;
    labelCanvas.height = 64;
    var ctx = labelCanvas.getContext('2d');
    ctx.fillStyle = '#1a1a1a';
    ctx.font = 'bold 36px Inter, Helvetica, Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(cityInfo.label, 256, 42);

    var texture = new THREE.CanvasTexture(labelCanvas);
    var spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
    var sprite = new THREE.Sprite(spriteMaterial);
    sprite.position.set(gridX, -12, gridZ);
    sprite.scale.set(80, 10, 1);
    scene.add(sprite);

    return mesh;
  }

  function updateAllColors() {
    CITIES.forEach(function (cityInfo) {
      var mesh = cityMeshes[cityInfo.name];
      var data = cityData[cityInfo.name];
      if (!mesh || !data) return;

      var voxels = data.voxels;
      var color = new THREE.Color();

      for (var i = 0; i < voxels.length; i++) {
        var rgb = getVoxelColor(voxels[i], currentMode, data.hasSolar);
        color.setRGB(rgb[0], rgb[1], rgb[2]);
        mesh.setColorAt(i, color);
      }
      mesh.instanceColor.needsUpdate = true;
    });
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
  }

  function stopAnimation() {
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  }

  function loadCity(cityInfo) {
    return fetch('assets/' + cityInfo.name + '_voxels.json')
      .then(function (res) {
        if (!res.ok) throw new Error('Failed: ' + cityInfo.name);
        return res.json();
      })
      .then(function (data) {
        cityData[cityInfo.name] = data;
        cityMeshes[cityInfo.name] = buildCityMesh(data, cityInfo);
        console.log('Loaded ' + cityInfo.name + ': ' + data.count + ' voxels');
      });
  }

  function initViewer() {
    if (isInitialized) return;
    if (!initScene()) return;
    isInitialized = true;

    // Load all 6 cities
    var promises = CITIES.map(function (c) {
      return loadCity(c).catch(function (err) {
        console.warn('VoxCity load failed for ' + c.name + ':', err);
      });
    });

    Promise.all(promises).then(function () {
      console.log('All cities loaded');
      animate();
    });

    // Toggle button
    var toggleBtn = document.getElementById('voxcity-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', function () {
        currentMode = currentMode === 'geometry' ? 'solar' : 'geometry';
        toggleBtn.dataset.mode = currentMode;
        var labels = toggleBtn.querySelectorAll('.btn-label, .btn-label-alt');
        if (labels.length >= 2) {
          labels[0].style.fontWeight = currentMode === 'geometry' ? '700' : '400';
          labels[0].style.color = currentMode === 'geometry' ? '#c4513a' : '#888888';
          labels[1].style.fontWeight = currentMode === 'solar' ? '700' : '400';
          labels[1].style.color = currentMode === 'solar' ? '#c4513a' : '#888888';
        }
        updateAllColors();
      });
    }

    window.addEventListener('resize', function () {
      var container = document.getElementById('voxcity-container');
      if (!container || !camera || !renderer) return;
      var w = container.clientWidth;
      var h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
  }

  // Reveal.js integration
  if (typeof Reveal !== 'undefined') {
    Reveal.on('slidechanged', function (event) {
      var container = event.currentSlide.querySelector('#voxcity-container');
      if (container && !isInitialized) initViewer();
      if (!container && animationId) stopAnimation();
      else if (container && isInitialized && !animationId) animate();
    });
  }
})();
