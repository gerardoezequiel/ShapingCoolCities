/**
 * Response Curve for 50% Regime Boundary — Chart.js
 *
 * Animated chart showing depaving % vs cooling °C with a clear
 * kink at the 50% regime boundary. Triggers when the slide appears.
 *
 * v3: Light theme — dark axis text, subtle grid, terracotta accents.
 */

(function () {
  'use strict';

  var chartInstance = null;
  var hasAnimated = false;
  var MAX_RETRIES = 30; // 3 seconds max wait for Chart.js
  var retryCount = 0;

  // Inline data fallback (in case JSON fails to load)
  var FALLBACK_DATA = {
    mean: [
      { x: 30, y: 1.14 },
      { x: 40, y: 1.17 },
      { x: 50, y: 1.22 }
    ],
    costEffective: [
      { x: 30, y: 1.13 },
      { x: 40, y: 1.16 },
      { x: 50, y: 1.20 }
    ],
    regimeBoundary: 50
  };

  function createChart(data) {
    var canvas = document.getElementById('responseCurveChart');
    if (!canvas) return;

    // Wait for Chart.js to load (may be deferred)
    if (typeof Chart === 'undefined') {
      if (retryCount < MAX_RETRIES) {
        retryCount++;
        setTimeout(function () { createChart(data); }, 100);
      }
      return;
    }

    // Prevent duplicate charts
    if (chartInstance) return;

    var ctx = canvas.getContext('2d');

    // Plugin: draw vertical dashed line at regime boundary
    var regimeBoundaryPlugin = {
      id: 'regimeBoundary',
      afterDraw: function (chart) {
        var xScale = chart.scales.x;
        var yScale = chart.scales.y;
        var ctx = chart.ctx;
        var xPos = xScale.getPixelForValue(data.regimeBoundary);

        ctx.save();
        ctx.strokeStyle = '#c4513a';
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(xPos, yScale.top);
        ctx.lineTo(xPos, yScale.bottom);
        ctx.stroke();

        // Label
        ctx.fillStyle = '#c4513a';
        ctx.font = 'bold 13px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Regime Boundary', xPos, yScale.top - 8);
        ctx.restore();
      }
    };

    // Build range band data (min-max area)
    var minData = data.min || [];
    var maxData = data.max || [];

    var datasets = [
      {
        label: 'Mean cooling (all combos)',
        data: data.mean,
        borderColor: '#3a6fa0',
        backgroundColor: 'rgba(58, 111, 160, 0.12)',
        borderWidth: 3,
        pointRadius: 5,
        pointBackgroundColor: '#3a6fa0',
        fill: false,
        tension: 0.3
      }
    ];

    if (data.costEffective && data.costEffective.length > 0) {
      datasets.push({
        label: 'Cost-effective (10% veg, 20% trees)',
        data: data.costEffective,
        borderColor: '#2a7f6f',
        backgroundColor: 'rgba(42, 127, 111, 0.12)',
        borderWidth: 3,
        pointRadius: 5,
        pointBackgroundColor: '#2a7f6f',
        fill: false,
        tension: 0.3,
        borderDash: [6, 3]
      });
    }

    // Max range band (faded)
    if (maxData.length > 0) {
      datasets.push({
        label: 'Max cooling',
        data: maxData,
        borderColor: 'rgba(58, 111, 160, 0.25)',
        backgroundColor: 'rgba(58, 111, 160, 0.06)',
        borderWidth: 1,
        pointRadius: 0,
        fill: '+1',
        tension: 0.3
      });
    }

    if (minData.length > 0) {
      datasets.push({
        label: 'Min cooling',
        data: minData,
        borderColor: 'rgba(58, 111, 160, 0.25)',
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        tension: 0.3
      });
    }

    chartInstance = new Chart(ctx, {
      type: 'line',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 1500,
          easing: 'easeOutQuart'
        },
        scales: {
          x: {
            type: 'linear',
            title: {
              display: true,
              text: 'Depaving %',
              color: '#555555',
              font: { family: 'Inter', size: 13 }
            },
            min: 25,
            max: 55,
            ticks: { color: '#555555', font: { family: 'Inter', size: 11 } },
            grid: { color: 'rgba(0,0,0,0.06)' }
          },
          y: {
            title: {
              display: true,
              text: 'Cooling (°C)',
              color: '#555555',
              font: { family: 'Inter', size: 13 }
            },
            ticks: { color: '#555555', font: { family: 'Inter', size: 11 } },
            grid: { color: 'rgba(0,0,0,0.06)' }
          }
        },
        plugins: {
          legend: {
            labels: {
              color: '#555555',
              font: { family: 'Inter', size: 11 },
              filter: function (item) {
                return item.text !== 'Max cooling' && item.text !== 'Min cooling';
              }
            }
          },
          tooltip: {
            backgroundColor: 'rgba(255,255,255,0.96)',
            titleColor: '#1a1a1a',
            bodyColor: '#555555',
            borderColor: 'rgba(0,0,0,0.08)',
            borderWidth: 1,
            titleFont: { family: 'Inter' },
            bodyFont: { family: 'Inter' },
            callbacks: {
              label: function (context) {
                return context.dataset.label + ': -' + context.parsed.y.toFixed(2) + '°C';
              }
            }
          }
        }
      },
      plugins: [regimeBoundaryPlugin]
    });
  }

  function initResponseCurve() {
    if (hasAnimated) return;
    hasAnimated = true;

    // Try loading JSON data
    fetch('assets/response_curve_data.json')
      .then(function (res) {
        if (!res.ok) throw new Error('Not found');
        return res.json();
      })
      .then(function (data) { createChart(data); })
      .catch(function () {
        console.warn('Response curve data not found, using fallback');
        createChart(FALLBACK_DATA);
      });
  }

  // Trigger chart when slide with the canvas becomes active
  function registerListeners() {
    if (typeof Reveal === 'undefined') return;

    Reveal.on('slidechanged', function (event) {
      var canvas = event.currentSlide.querySelector('#responseCurveChart');
      if (canvas && !hasAnimated) {
        initResponseCurve();
      }
    });

    // Also check fragments (in case canvas is inside one)
    Reveal.on('fragmentshown', function (event) {
      var canvas = event.fragment.querySelector && event.fragment.querySelector('#responseCurveChart');
      if (!canvas) {
        canvas = event.fragment.closest && event.fragment.closest('section') &&
          event.fragment.closest('section').querySelector('#responseCurveChart');
      }
      if (canvas && !hasAnimated) {
        initResponseCurve();
      }
    });

    // Fallback: check if already on the slide (e.g., direct hash navigation)
    Reveal.on('ready', function () {
      var currentSlide = Reveal.getCurrentSlide();
      if (currentSlide && currentSlide.querySelector('#responseCurveChart') && !hasAnimated) {
        initResponseCurve();
      }
    });
  }

  registerListeners();
})();
