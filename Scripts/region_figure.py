"""Plotly HTML events for masks that appear only over a SAM region."""
from pathlib import Path
from html import escape
import tempfile
import webbrowser

from plotly.graph_objects import Figure, Image as ImageTrace

MASK_HOVER_SCRIPT = """
(function () {
    const graph = document.getElementById('{plot_id}');
    let shown = null;
    let pointer = null;
    function selectMask(next) {
        if (next === shown) return;
        const update = {};
        if (shown !== null) update['images[' + shown + '].visible'] = false;
        if (next !== null) update['images[' + next + '].visible'] = true;
        shown = next;
        Plotly.relayout(graph, update);
    }
    function updateMask(event) {
        pointer = event;
        const layout = graph._fullLayout;
        const rect = graph.getBoundingClientRect();
        if (!layout || !rect.width || !rect.height || event.buttons) {
            selectMask(null);
            return;
        }
        const px = (event.clientX - rect.left) * layout.width / rect.width;
        const py = (event.clientY - rect.top) * layout.height / rect.height;
        let best = null;
        let area = Infinity;
        for (const trace of graph.data) {
            const meta = trace.meta;
            if (!meta || !Number.isInteger(meta.sam_mask_index) || !Array.isArray(meta.sam_bbox)) continue;
            const xa = layout['xaxis' + (trace.xaxis || 'x').slice(1)];
            const ya = layout['yaxis' + (trace.yaxis || 'y').slice(1)];
            if (!xa || !ya || px < xa._offset || px > xa._offset + xa._length ||
                py < ya._offset || py > ya._offset + ya._length) continue;
            const x = xa.p2d(px - xa._offset);
            const y = ya.p2d(py - ya._offset);
            const [x1, y1, x2, y2] = meta.sam_bbox;
            const candidateArea = (x2 - x1) * (y2 - y1);
            if (x >= x1 && x < x2 && y >= y1 && y < y2 && candidateArea > 0 &&
                (candidateArea < area || (candidateArea === area && meta.sam_mask_index < best))) {
                area = candidateArea;
                best = meta.sam_mask_index;
            }
        }
        selectMask(best);
    }
    graph.addEventListener('mousemove', updateMask, true);
    graph.addEventListener('mouseup', updateMask, true);
    graph.addEventListener('mouseleave', function () { pointer = null; selectMask(null); });
    graph.on('plotly_relayout', function () { if (pointer) updateMask(pointer); });
})();
"""


def _with_mask_events(kwargs):
    options = dict(kwargs)
    extra = options.pop("post_script", None)
    scripts = [MASK_HOVER_SCRIPT]
    if isinstance(extra, str):
        scripts.append(extra)
    elif extra:
        scripts.extend(extra)
    options["post_script"] = scripts
    return options


class HoverMaskFigure(Figure):
    """Retain callbacks in notebook display, show(), and exported HTML."""

    def to_html(self, *args, **kwargs):
        return super().to_html(*args, **_with_mask_events(kwargs))

    def write_html(self, *args, **kwargs):
        return super().write_html(*args, **_with_mask_events(kwargs))

    def _notebook_html(self, *, validate=True, **kwargs):
        """Isolate Plotly from the notebook's renderer and retain a PNG fallback."""
        height = self.layout.height or kwargs.get("default_height", 600)
        css_height = f"{height}px" if isinstance(height, (int, float)) else str(height)
        document = self.to_html(full_html=True, include_plotlyjs=True, validate=validate, **kwargs)
        document = document.replace("<head>", "<head><style>html,body{margin:0;background:transparent;}</style>", 1)
        source = next((trace.source for trace in self.data
                       if isinstance(trace, ImageTrace) and isinstance(trace.source, str)), None)
        fallback = (f'<img src="{escape(source, quote=True)}" alt="Исходное изображение" '
                    'style="display:block;width:100%;height:100%;object-fit:contain;">') if source else ""
        return (
            f'<div style="position:relative;width:100%;height:{escape(css_height, quote=True)};">'
            f'{fallback}'
            '<iframe title="Области изображения" sandbox="allow-scripts allow-downloads" '
            'style="position:absolute;inset:0;width:100%;height:100%;border:0;background:transparent;" '
            f'srcdoc="{escape(document, quote=True)}"></iframe></div>'
        )

    def _repr_mimebundle_(self, include=None, exclude=None, validate=True, **kwargs):
        return {"text/html": self._notebook_html(validate=validate)}

    def _ipython_display_(self):
        self.show()

    def show(self, renderer=None, validate=True, **kwargs):
        if renderer in ("png", "jpeg", "jpg", "svg", "pdf"):
            return super().show(renderer=renderer, validate=validate, **kwargs)
        from IPython.core.getipython import get_ipython
        shell = get_ipython()
        notebook = renderer != "browser" and shell is not None and hasattr(shell, "kernel")
        kwargs.setdefault("default_height", 600)
        if notebook:
            from IPython.core.display import HTML
            from IPython.display import display
            display(HTML(self._notebook_html(validate=validate, **kwargs)))
        else:
            html = self.to_html(full_html=True, include_plotlyjs=True, validate=validate, **kwargs)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".html", delete=False) as stream:
                stream.write(html)
                path = Path(stream.name)
            webbrowser.open_new_tab(path.as_uri())
        return None
