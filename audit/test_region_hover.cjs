// Run the actual hover handler without requiring a notebook or GPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const python = fs.readFileSync(path.join(__dirname, '../Scripts/region_figure.py'), 'utf8');
const script = python.split('MASK_HOVER_SCRIPT = """')[1].split('"""')[0];

function region(index, bbox, extra = {}) {
    return {meta: {sam_mask_index: index, sam_bbox: bbox}, ...extra};
}

function fixture(traces) {
    const handlers = {};
    const visible = new Map();
    const changes = [];
    const graph = {
        data: traces,
        _fullLayout: {
            width: 400, height: 300,
            xaxis: {_offset: 20, _length: 300, p2d: value => value},
            yaxis: {_offset: 30, _length: 200, p2d: value => value},
        },
        rect: {left: 10, top: 15, width: 400, height: 300},
        getBoundingClientRect() { return this.rect; },
        addEventListener(name, handler) { handlers[name] = handler; },
        on(name, handler) { handlers[name] = handler; },
    };
    vm.runInNewContext(script, {
        document: {getElementById() { return graph; }},
        Plotly: {relayout(target, update) {
            assert.equal(target, graph);
            changes.push(update);
            for (const [key, value] of Object.entries(update)) {
                visible.set(Number(key.match(/images\[(\d+)\]/)[1]), value);
            }
            handlers.plotly_relayout();
        }},
    });
    function move(px, py, buttons = 0) {
        handlers.mousemove({
            clientX: graph.rect.left + px * graph.rect.width / graph._fullLayout.width,
            clientY: graph.rect.top + py * graph.rect.height / graph._fullLayout.height,
            buttons,
        });
    }
    return {graph, handlers, changes, move,
        selected: () => [...visible].filter(([, shown]) => shown).map(([index]) => index)};
}

for (const reversed of [false, true]) {
    test(`smallest nested box wins regardless of trace order (${reversed})`, () => {
        const traces = [region(0, [0, 0, 250, 180]), region(1, [40, 40, 100, 100]),
            region(2, [50, 50, 60, 60]), {type: 'image'}, {meta: {source: 'yolo'}}];
        const f = fixture(reversed ? traces.reverse() : traces);
        assert.deepEqual(f.selected(), []);
        f.move(25, 35); assert.deepEqual(f.selected(), [0]);
        f.move(65, 75); assert.deepEqual(f.selected(), [1]);
        f.move(75, 85); assert.deepEqual(f.selected(), [2]);
        f.move(65, 75); assert.deepEqual(f.selected(), [1]);
        f.move(25, 35); assert.deepEqual(f.selected(), [0]);
        f.move(300, 220); assert.deepEqual(f.selected(), []);
    });
}

test('partially overlapping boxes use area, with deterministic ties', () => {
    const f = fixture([region(8, [0, 0, 100, 100]), region(3, [50, 50, 130, 130]),
        region(2, [50, 50, 130, 130])]);
    f.move(95, 105); assert.deepEqual(f.selected(), [2]);
    f.move(25, 35); assert.deepEqual(f.selected(), [8]);
    f.move(120, 130); assert.deepEqual(f.selected(), [2]);
});

test('zoom, reversed Y, CSS scaling and viewport clipping', () => {
    const f = fixture([region(0, [0, 0, 400, 300]), region(1, [100, 40, 120, 60])]);
    f.graph.rect.width = 800;
    f.graph.rect.height = 150;
    f.graph._fullLayout.xaxis.p2d = value => value / 2 + 90;
    f.graph._fullLayout.yaxis.p2d = value => 100 - value / 2;
    f.move(60, 130); assert.deepEqual(f.selected(), [1]);
    f.move(19, 130); assert.deepEqual(f.selected(), []);
    f.move(60, 29); assert.deepEqual(f.selected(), []);
    f.move(321, 130); assert.deepEqual(f.selected(), []);
    f.move(60, 231); assert.deepEqual(f.selected(), []);
});

test('inspector crop subplot does not activate masks from the source subplot', () => {
    const f = fixture([region(0, [0, 0, 300, 200])]);
    f.graph._fullLayout.xaxis._length = 150;
    f.graph._fullLayout.xaxis2 = {_offset: 200, _length: 150, p2d: value => value};
    f.move(50, 60); assert.deepEqual(f.selected(), [0]);
    f.move(220, 60); assert.deepEqual(f.selected(), []);
});

test('pointer leave and dragging hide masks, and repeated moves do not redraw', () => {
    const f = fixture([region(0, [0, 0, 100, 100])]);
    f.move(50, 60); f.move(51, 61);
    assert.equal(f.changes.length, 1);
    f.move(50, 60, 1); assert.deepEqual(f.selected(), []);
    f.move(50, 60); assert.deepEqual(f.selected(), [0]);
    f.handlers.mouseleave(); assert.deepEqual(f.selected(), []);
    f.handlers.plotly_relayout(); assert.deepEqual(f.selected(), []);
});

test('relayout recomputes the hit under a stationary pointer', () => {
    const f = fixture([region(0, [0, 0, 250, 180]), region(1, [40, 40, 100, 100])]);
    f.move(65, 75); assert.deepEqual(f.selected(), [1]);
    f.graph._fullLayout.xaxis.p2d = value => value + 100;
    f.handlers.plotly_relayout(); assert.deepEqual(f.selected(), [0]);
});
