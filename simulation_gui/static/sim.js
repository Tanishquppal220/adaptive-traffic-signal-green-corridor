/**
 * Simulation GUI JavaScript
 * 
 * Connects to /api/state SSE stream, receives SimSnapshot every second,
 * and updates all DOM/SVG elements.
 */

// Constants
const DIRECTIONS = ['N', 'S', 'E', 'W'];
const DIRECTION_FULL = { N: 'North', S: 'South', E: 'East', W: 'West' };
const SVG_NS = 'http://www.w3.org/2000/svg';

// Color palette
const COLORS = {
    green: '#22c55e',
    yellow: '#eab308',
    red: '#ef4444',
    road: '#1a1a2e',
    roadLight: '#252540',
    marking: '#ffffff',
    vehicle: '#60a5fa',
    ambulance: '#ef4444',
    corridorGlow: 'rgba(34, 197, 94, 0.3)',
};

// Road geometry (600x600 viewBox)
const GEOMETRY = {
    roadWidth: 120,
    intersectionSize: 120,
    intersectionStart: 240,
    intersectionEnd: 360,
    vehicleWidth: 8,
    vehicleHeight: 14,
    vehicleGap: 3,
    maxVisibleVehicles: 8,
};

/**
 * SSE Client - connects to /api/state and dispatches updates
 */
class SimClient {
    constructor(onSnapshot) {
        this.onSnapshot = onSnapshot;
        this.es = null;
        this.reconnectDelay = 1000;
    }

    connect() {
        this.es = new EventSource('/api/state');
        
        this.es.onmessage = (e) => {
            try {
                const snapshot = JSON.parse(e.data);
                this.onSnapshot(snapshot);
            } catch (err) {
                console.error('Parse error:', err);
            }
        };

        this.es.onerror = () => {
            console.warn('SSE connection lost, reconnecting...');
            this.es.close();
            setTimeout(() => this.connect(), this.reconnectDelay);
        };
    }
}

/**
 * IntersectionRenderer - handles all SVG drawing and updates
 */
class IntersectionRenderer {
    constructor(svgEl) {
        this.svg = svgEl;
        this.elements = {
            vehicles: {},
            vehicleCounts: {},
            signals: {},
            ambulance: null,
            corridorOverlay: null,
        };
        this._buildStaticElements();
    }

    _createSvgElement(tag, attrs = {}) {
        const el = document.createElementNS(SVG_NS, tag);
        for (const [key, value] of Object.entries(attrs)) {
            el.setAttribute(key, value);
        }
        return el;
    }

    _buildStaticElements() {
        // Clear existing
        this.svg.innerHTML = '';

        // Background
        const bg = this._createSvgElement('rect', {
            x: 0, y: 0, width: 600, height: 600,
            fill: '#0a0e1a'
        });
        this.svg.appendChild(bg);

        // Roads
        // N-S road
        const nsRoad = this._createSvgElement('rect', {
            x: GEOMETRY.intersectionStart, y: 0,
            width: GEOMETRY.roadWidth, height: 600,
            fill: COLORS.road
        });
        this.svg.appendChild(nsRoad);

        // E-W road
        const ewRoad = this._createSvgElement('rect', {
            x: 0, y: GEOMETRY.intersectionStart,
            width: 600, height: GEOMETRY.roadWidth,
            fill: COLORS.road
        });
        this.svg.appendChild(ewRoad);

        // Intersection box (slightly lighter)
        const intersection = this._createSvgElement('rect', {
            x: GEOMETRY.intersectionStart, y: GEOMETRY.intersectionStart,
            width: GEOMETRY.intersectionSize, height: GEOMETRY.intersectionSize,
            fill: COLORS.roadLight
        });
        this.svg.appendChild(intersection);

        // Lane markings (center lines)
        this._drawLaneMarkings();

        // Stop lines
        this._drawStopLines();

        // Traffic light housings
        this._buildTrafficLights();

        // Corridor overlay (initially hidden)
        this._buildCorridorOverlays();

        // Vehicle containers
        this._buildVehicleContainers();

        // Ambulance element (initially hidden)
        this._buildAmbulance();
    }

    _drawLaneMarkings() {
        // N-S center dashed line
        const dashLength = 20;
        const gapLength = 15;
        for (let y = 0; y < 600; y += dashLength + gapLength) {
            if (y >= GEOMETRY.intersectionStart - 10 && y <= GEOMETRY.intersectionEnd + 10) continue;
            const line = this._createSvgElement('rect', {
                x: 298, y: y, width: 4, height: dashLength,
                fill: COLORS.marking, opacity: 0.6
            });
            this.svg.appendChild(line);
        }

        // E-W center dashed line
        for (let x = 0; x < 600; x += dashLength + gapLength) {
            if (x >= GEOMETRY.intersectionStart - 10 && x <= GEOMETRY.intersectionEnd + 10) continue;
            const line = this._createSvgElement('rect', {
                x: x, y: 298, width: dashLength, height: 4,
                fill: COLORS.marking, opacity: 0.6
            });
            this.svg.appendChild(line);
        }
    }

    _drawStopLines() {
        // North stop line (vehicles from N stop here)
        this.svg.appendChild(this._createSvgElement('rect', {
            x: GEOMETRY.intersectionStart, y: GEOMETRY.intersectionStart - 6,
            width: GEOMETRY.roadWidth, height: 4,
            fill: COLORS.marking, opacity: 0.8
        }));

        // South stop line
        this.svg.appendChild(this._createSvgElement('rect', {
            x: GEOMETRY.intersectionStart, y: GEOMETRY.intersectionEnd + 2,
            width: GEOMETRY.roadWidth, height: 4,
            fill: COLORS.marking, opacity: 0.8
        }));

        // East stop line
        this.svg.appendChild(this._createSvgElement('rect', {
            x: GEOMETRY.intersectionEnd + 2, y: GEOMETRY.intersectionStart,
            width: 4, height: GEOMETRY.roadWidth,
            fill: COLORS.marking, opacity: 0.8
        }));

        // West stop line
        this.svg.appendChild(this._createSvgElement('rect', {
            x: GEOMETRY.intersectionStart - 6, y: GEOMETRY.intersectionStart,
            width: 4, height: GEOMETRY.roadWidth,
            fill: COLORS.marking, opacity: 0.8
        }));
    }

    _buildTrafficLights() {
        // Traffic light positions (housing position)
        const positions = {
            N: { x: 282, y: 195, vertical: true },   // Above intersection
            S: { x: 282, y: 365, vertical: true },   // Below intersection
            E: { x: 365, y: 282, vertical: false },  // Right of intersection
            W: { x: 195, y: 282, vertical: false },  // Left of intersection
        };

        this.elements.signals = {};

        for (const dir of DIRECTIONS) {
            const pos = positions[dir];
            const group = this._createSvgElement('g');
            
            // Housing
            const housingWidth = pos.vertical ? 24 : 56;
            const housingHeight = pos.vertical ? 56 : 24;
            const housing = this._createSvgElement('rect', {
                x: pos.x, y: pos.y,
                width: housingWidth, height: housingHeight,
                fill: '#1f2937', rx: 4, ry: 4,
                stroke: '#374151', 'stroke-width': 1
            });
            group.appendChild(housing);

            // Light circles
            const lights = {};
            const colors = ['red', 'yellow', 'green'];
            
            for (let i = 0; i < 3; i++) {
                const cx = pos.vertical ? pos.x + 12 : pos.x + 10 + i * 18;
                const cy = pos.vertical ? pos.y + 10 + i * 18 : pos.y + 12;
                
                const circle = this._createSvgElement('circle', {
                    cx: cx, cy: cy, r: 7,
                    fill: '#333333', opacity: 0.3
                });
                group.appendChild(circle);
                lights[colors[i]] = circle;
            }

            this.svg.appendChild(group);
            this.elements.signals[dir] = lights;
        }
    }

    _buildCorridorOverlays() {
        this.elements.corridorOverlays = {};
        
        const overlayDefs = {
            N: { x: GEOMETRY.intersectionStart, y: 0, width: GEOMETRY.roadWidth, height: GEOMETRY.intersectionStart },
            S: { x: GEOMETRY.intersectionStart, y: GEOMETRY.intersectionEnd, width: GEOMETRY.roadWidth, height: 600 - GEOMETRY.intersectionEnd },
            E: { x: GEOMETRY.intersectionEnd, y: GEOMETRY.intersectionStart, width: 600 - GEOMETRY.intersectionEnd, height: GEOMETRY.roadWidth },
            W: { x: 0, y: GEOMETRY.intersectionStart, width: GEOMETRY.intersectionStart, height: GEOMETRY.roadWidth },
        };

        for (const dir of DIRECTIONS) {
            const def = overlayDefs[dir];
            const overlay = this._createSvgElement('rect', {
                ...def,
                fill: COLORS.corridorGlow,
                opacity: 0,
                class: 'corridor-overlay'
            });
            this.svg.appendChild(overlay);
            this.elements.corridorOverlays[dir] = overlay;
        }

        // Add CSS animation for pulsing
        const style = document.createElement('style');
        style.textContent = `
            .corridor-overlay.active {
                animation: corridorPulse 1s infinite ease-in-out;
            }
            @keyframes corridorPulse {
                0%, 100% { opacity: 0.2; }
                50% { opacity: 0.5; }
            }
        `;
        document.head.appendChild(style);
    }

    _buildVehicleContainers() {
        // Vehicle positions (starting point for queue)
        const vehicleStartPositions = {
            N: { x: 270, startY: GEOMETRY.intersectionStart - 20, dir: -1 },  // Stack upward
            S: { x: 310, startY: GEOMETRY.intersectionEnd + 20, dir: 1 },     // Stack downward
            E: { x: GEOMETRY.intersectionEnd + 20, y: 310, dir: 1 },          // Stack rightward
            W: { x: GEOMETRY.intersectionStart - 20, y: 270, dir: -1 },       // Stack leftward
        };

        for (const dir of DIRECTIONS) {
            const container = this._createSvgElement('g', { class: `vehicles-${dir}` });
            this.svg.appendChild(container);
            this.elements.vehicles[dir] = {
                container,
                rects: [],
                position: vehicleStartPositions[dir]
            };

            // Vehicle count label
            const pos = vehicleStartPositions[dir];
            let labelX, labelY;
            if (dir === 'N') { labelX = pos.x + 30; labelY = pos.startY - 40; }
            else if (dir === 'S') { labelX = pos.x + 30; labelY = pos.startY + 60; }
            else if (dir === 'E') { labelX = pos.x + 50; labelY = pos.y + 5; }
            else { labelX = pos.x - 50; labelY = pos.y + 5; }

            const label = this._createSvgElement('text', {
                x: labelX, y: labelY,
                fill: COLORS.vehicle,
                'font-size': '14px',
                'font-weight': 'bold',
                'text-anchor': 'middle'
            });
            this.svg.appendChild(label);
            this.elements.vehicleCounts[dir] = label;
        }
    }

    _buildAmbulance() {
        const group = this._createSvgElement('g', {
            class: 'ambulance',
            style: 'display: none;'
        });

        // Ambulance body
        const body = this._createSvgElement('rect', {
            x: -9, y: -14, width: 18, height: 28,
            fill: COLORS.ambulance, rx: 3, ry: 3,
            stroke: '#ffffff', 'stroke-width': 2
        });
        group.appendChild(body);

        // Cross symbol
        const crossH = this._createSvgElement('rect', {
            x: -5, y: -2, width: 10, height: 4,
            fill: '#ffffff'
        });
        const crossV = this._createSvgElement('rect', {
            x: -2, y: -5, width: 4, height: 10,
            fill: '#ffffff'
        });
        group.appendChild(crossH);
        group.appendChild(crossV);

        this.svg.appendChild(group);
        this.elements.ambulance = group;
    }

    update(snapshot) {
        this._updateSignals(snapshot.signals);
        this._updateVehicles(snapshot.lanes);
        this._updateAmbulance(snapshot.ambulance);
        this._updateCorridorHighlight(snapshot.ambulance);
    }

    _updateSignals(signals) {
        for (const dir of DIRECTIONS) {
            const sigState = signals[dir];
            const lights = this.elements.signals[dir];
            
            // Reset all lights to dim
            lights.red.setAttribute('fill', '#333333');
            lights.red.setAttribute('opacity', '0.3');
            lights.yellow.setAttribute('fill', '#333333');
            lights.yellow.setAttribute('opacity', '0.3');
            lights.green.setAttribute('fill', '#333333');
            lights.green.setAttribute('opacity', '0.3');

            // Activate current color
            const activeColor = sigState.color.toLowerCase();
            if (lights[activeColor]) {
                lights[activeColor].setAttribute('fill', COLORS[activeColor]);
                lights[activeColor].setAttribute('opacity', '1');
            }
        }
    }

    _updateVehicles(lanes) {
        for (const dir of DIRECTIONS) {
            const lane = lanes[dir];
            const vehicleData = this.elements.vehicles[dir];
            const container = vehicleData.container;
            const pos = vehicleData.position;
            const count = lane.vehicle_count;

            // Update count label
            this.elements.vehicleCounts[dir].textContent = count > 0 ? count : '';

            // Clear existing vehicle rects
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }

            // Draw up to maxVisibleVehicles
            const visibleCount = Math.min(count, GEOMETRY.maxVisibleVehicles);

            for (let i = 0; i < visibleCount; i++) {
                let rect;
                const offset = i * (GEOMETRY.vehicleHeight + GEOMETRY.vehicleGap);

                if (dir === 'N') {
                    rect = this._createSvgElement('rect', {
                        x: pos.x, y: pos.startY - offset - GEOMETRY.vehicleHeight,
                        width: GEOMETRY.vehicleWidth, height: GEOMETRY.vehicleHeight,
                        fill: COLORS.vehicle, rx: 2, ry: 2,
                        opacity: 1 - (i * 0.08)
                    });
                } else if (dir === 'S') {
                    rect = this._createSvgElement('rect', {
                        x: pos.x, y: pos.startY + offset,
                        width: GEOMETRY.vehicleWidth, height: GEOMETRY.vehicleHeight,
                        fill: COLORS.vehicle, rx: 2, ry: 2,
                        opacity: 1 - (i * 0.08)
                    });
                } else if (dir === 'E') {
                    rect = this._createSvgElement('rect', {
                        x: pos.x + offset, y: pos.y,
                        width: GEOMETRY.vehicleHeight, height: GEOMETRY.vehicleWidth,
                        fill: COLORS.vehicle, rx: 2, ry: 2,
                        opacity: 1 - (i * 0.08)
                    });
                } else { // W
                    rect = this._createSvgElement('rect', {
                        x: pos.x - offset - GEOMETRY.vehicleHeight, y: pos.y,
                        width: GEOMETRY.vehicleHeight, height: GEOMETRY.vehicleWidth,
                        fill: COLORS.vehicle, rx: 2, ry: 2,
                        opacity: 1 - (i * 0.08)
                    });
                }

                container.appendChild(rect);
            }

            // Show "+N more" if count exceeds visible
            if (count > GEOMETRY.maxVisibleVehicles) {
                const moreCount = count - GEOMETRY.maxVisibleVehicles;
                this.elements.vehicleCounts[dir].textContent = `${count} (+${moreCount})`;
            }
        }
    }

    _updateAmbulance(ambulance) {
        const group = this.elements.ambulance;

        if (!ambulance.active) {
            group.style.display = 'none';
            return;
        }

        group.style.display = 'block';

        // Calculate position based on entry direction and position value
        const pos = ambulance.position;
        const entry = ambulance.entry_direction || 'N';
        const exit = ambulance.exit_direction;

        let x, y, rotation = 0;

        // Entry positions
        const entryCoords = {
            N: { x: 280, y: 50 + (pos * 250) },
            S: { x: 320, y: 550 - (pos * 250) },
            E: { x: 550 - (pos * 250), y: 320 },
            W: { x: 50 + (pos * 250), y: 280 },
        };

        // Exit positions (after crossing intersection)
        const exitCoords = {
            N: { x: 280, y: 300 - ((pos - 1) * 250) },
            S: { x: 320, y: 300 + ((pos - 1) * 250) },
            E: { x: 300 + ((pos - 1) * 250), y: 320 },
            W: { x: 300 - ((pos - 1) * 250), y: 280 },
        };

        if (pos < 1.0) {
            // Approaching
            const coords = entryCoords[entry];
            x = coords.x;
            y = coords.y;
            rotation = entry === 'N' ? 180 : entry === 'S' ? 0 : entry === 'E' ? 270 : 90;
        } else if (exit) {
            // Exiting
            const coords = exitCoords[exit];
            x = coords.x;
            y = coords.y;
            rotation = exit === 'N' ? 0 : exit === 'S' ? 180 : exit === 'E' ? 90 : 270;
        } else {
            // At intersection center
            x = 300;
            y = 300;
        }

        group.setAttribute('transform', `translate(${x}, ${y}) rotate(${rotation})`);
    }

    _updateCorridorHighlight(ambulance) {
        // Hide all overlays first
        for (const dir of DIRECTIONS) {
            this.elements.corridorOverlays[dir].classList.remove('active');
            this.elements.corridorOverlays[dir].style.opacity = '0';
        }

        if (ambulance.corridor_active && ambulance.corridor_direction) {
            const overlay = this.elements.corridorOverlays[ambulance.corridor_direction];
            if (overlay) {
                overlay.classList.add('active');
            }
        }
    }
}

/**
 * MetricsPanel - updates right panel DOM elements
 */
class MetricsPanel {
    update(snapshot) {
        // Update density bars
        for (const dir of DIRECTIONS) {
            const count = snapshot.lanes[dir].vehicle_count;
            const percentage = Math.min(100, (count / 50) * 100);
            
            document.getElementById(`density-${dir}`).style.width = `${percentage}%`;
            document.getElementById(`count-${dir}`).textContent = count;
        }

        // Update signal timers
        for (const dir of DIRECTIONS) {
            const sig = snapshot.signals[dir];
            document.getElementById(`time-${dir}`).textContent = sig.time_remaining;
            
            const dot = document.getElementById(`dot-${dir}`);
            dot.className = 'timer-dot ' + sig.color.toLowerCase();
        }

        // Update mode badge
        const modeBadge = document.getElementById('mode-badge');
        modeBadge.textContent = snapshot.mode;
        modeBadge.className = 'mode-badge ' + snapshot.mode.toLowerCase().replace('_', '-');
        if (snapshot.mode === 'CORRIDOR_ACTIVE') {
            modeBadge.className = 'mode-badge corridor';
        }

        // Update statistics
        document.getElementById('stat-passed').textContent = snapshot.metrics.total_vehicles_passed;
        document.getElementById('stat-wait').textContent = snapshot.metrics.avg_wait_time + 's';
        document.getElementById('stat-throughput').textContent = snapshot.metrics.throughput_per_minute + '/min';
        document.getElementById('stat-speed').textContent = snapshot.sim_speed + 'x';

        // Sync speed buttons with actual state
        const speedButtons = document.querySelectorAll('.speed-buttons button');
        speedButtons.forEach(btn => {
            const btnSpeed = parseFloat(btn.dataset.speed);
            if (btnSpeed === snapshot.sim_speed) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update emergency section
        const emergencySection = document.getElementById('emergency-section');
        if (snapshot.ambulance.active) {
            emergencySection.style.display = 'block';
            document.getElementById('detection-visual').textContent = 
                snapshot.ambulance.confidence > 0 
                    ? (snapshot.ambulance.confidence * 100).toFixed(0) + '%' 
                    : 'Detecting...';
            document.getElementById('detection-entry').textContent = 
                DIRECTION_FULL[snapshot.ambulance.entry_direction] || '--';
            document.getElementById('detection-corridor').textContent = 
                snapshot.ambulance.corridor_direction 
                    ? DIRECTION_FULL[snapshot.ambulance.corridor_direction] + ' (Active)'
                    : 'Pending...';
        } else {
            emergencySection.style.display = 'none';
        }

        // Update ML predictions section
        const mlStatus = document.getElementById('ml-status');
        const mlPredictions = document.getElementById('ml-predictions');
        
        if (snapshot.ml && snapshot.ml.available) {
            mlStatus.textContent = snapshot.ml.history_ready 
                ? '✓ Active - Using predictions for signal timing'
                : '⏳ Warming up... (collecting history)';
            mlStatus.className = 'ml-status ' + (snapshot.ml.history_ready ? 'active' : '');
            
            if (snapshot.ml.predictions && Object.keys(snapshot.ml.predictions).length > 0) {
                mlPredictions.style.display = 'block';
                
                for (const dir of DIRECTIONS) {
                    const current = snapshot.lanes[dir].vehicle_count;
                    const predicted = snapshot.ml.predictions[dir] || current;
                    const change = predicted - current;
                    
                    document.getElementById(`pred-curr-${dir}`).textContent = current;
                    const futEl = document.getElementById(`pred-fut-${dir}`);
                    futEl.textContent = Math.round(predicted);
                    
                    // Color code based on change
                    futEl.className = 'pred-future';
                    if (change > 2) futEl.classList.add('increase');
                    else if (change < -2) futEl.classList.add('decrease');
                    else futEl.classList.add('stable');
                }
            } else {
                mlPredictions.style.display = 'none';
            }
        } else {
            mlStatus.textContent = '✗ ML models not loaded';
            mlStatus.className = 'ml-status inactive';
            mlPredictions.style.display = 'none';
        }

        // Update auto-gen toggle state
        document.getElementById('auto-gen-toggle').checked = snapshot.auto_generate;
    }
}

/**
 * EventLog - manages event log display
 */
class EventLog {
    constructor(containerEl) {
        this.container = containerEl;
        this.lastEvents = [];
    }

    update(events) {
        // Find new events
        const newEvents = events.filter(e => !this.lastEvents.includes(e));
        
        for (const event of newEvents) {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = event;

            // Color coding by emoji prefix
            if (event.includes('🚑') || event.includes('🚨')) {
                entry.classList.add('amber');
            } else if (event.includes('🟢') || event.includes('✅')) {
                entry.classList.add('green');
            } else if (event.includes('❌')) {
                entry.classList.add('red');
            } else if (event.includes('📡') || event.includes('🚦')) {
                entry.classList.add('blue');
            }

            this.container.appendChild(entry);
        }

        // Auto-scroll to bottom
        this.container.scrollTop = this.container.scrollHeight;

        // Trim if too many entries
        while (this.container.children.length > 50) {
            this.container.removeChild(this.container.firstChild);
        }

        this.lastEvents = [...events];
    }
}

/**
 * ControlPanel - handles user interactions
 */
class ControlPanel {
    constructor() {
        this._bindScenario();
        this._bindSpeed();
        this._bindAutoGen();
        this._bindLaneInputs();
        this._bindAmbulanceButtons();
        this._bindReset();
    }

    _post(endpoint, body) {
        return fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    }

    _bindScenario() {
        const select = document.getElementById('scenario-select');
        select.addEventListener('change', () => {
            this._post('/api/set_scenario', { scenario: select.value });
        });
    }

    _bindSpeed() {
        const buttons = document.querySelectorAll('.speed-buttons button');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                const speed = parseFloat(btn.dataset.speed);
                this._post('/api/set_speed', { speed });
                
                // Update active state
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    }

    _bindAutoGen() {
        const toggle = document.getElementById('auto-gen-toggle');
        toggle.addEventListener('change', () => {
            this._post('/api/toggle_auto_gen', { enabled: toggle.checked });
        });
    }

    _bindLaneInputs() {
        const applyBtn = document.getElementById('apply-counts-btn');
        applyBtn.addEventListener('click', () => {
            const counts = {
                N: parseInt(document.getElementById('lane-N').value) || 0,
                S: parseInt(document.getElementById('lane-S').value) || 0,
                E: parseInt(document.getElementById('lane-E').value) || 0,
                W: parseInt(document.getElementById('lane-W').value) || 0,
            };
            this._post('/api/set_lane_counts', counts);
        });
    }

    _bindAmbulanceButtons() {
        const buttons = document.querySelectorAll('.ambulance-buttons button');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                const direction = btn.dataset.direction;
                this._post('/api/trigger_ambulance', { entry: direction });
                
                // Disable button temporarily
                btn.disabled = true;
                setTimeout(() => { btn.disabled = false; }, 5000);
            });
        });
    }

    _bindReset() {
        const resetBtn = document.getElementById('reset-btn');
        resetBtn.addEventListener('click', () => {
            this._post('/api/reset', {});
        });
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const svg = document.getElementById('intersection-svg');
    const renderer = new IntersectionRenderer(svg);
    const metrics = new MetricsPanel();
    const log = new EventLog(document.getElementById('event-log'));
    const controls = new ControlPanel();

    const client = new SimClient((snapshot) => {
        renderer.update(snapshot);
        metrics.update(snapshot);
        log.update(snapshot.event_log);
    });

    client.connect();
});
