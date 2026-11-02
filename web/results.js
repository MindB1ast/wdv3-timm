import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { imagePoint, regionsAt, nextRegion } from "./region_selection.mjs";
import { repairPreviewLimit } from "./preview_limit.mjs";
import { mergedTagOrigin } from "./tag_origin.mjs";

const element = (tag, text) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
};

function regionCanvas(row, regions, pick) {
    const wrapper = element("div");
    const canvas = element("canvas");
    canvas.className = "wdv3-region-canvas";
    canvas.setAttribute("role", "img");
    canvas.tabIndex = 0;
    const status = element("div");
    status.setAttribute("role", "status");
    const hint = element("div", "Click a region to select it. Click again to cycle overlaps. ← / → select; Esc resets.");
    hint.style.cssText = "font-size:11px;color:#bbb;margin:6px 0";
    wrapper.append(canvas, status, hint);
    const context = canvas.getContext("2d");
    const base = new Image();
    const masks = new Map();
    let selected = null, lastClick = null, shapeKey = null, shape = null;
    let width, height;
    const imageURL = descriptor => api.apiURL(`/view?${new URLSearchParams(descriptor)}`);
    const containsMask = (region, x, y) => {
        const bits = masks.get(region.key);
        if (!bits) return true;
        const px = Math.min(canvas.width - 1, Math.floor(x * canvas.width / width));
        const py = Math.min(canvas.height - 1, Math.floor(y * canvas.height / height));
        const index = py * canvas.width + px;
        return Boolean(bits[index >> 3] & (1 << (index & 7)));
    };
    const drawBox = (region, active) => {
        const [x1, y1, x2, y2] = region.bbox;
        context.strokeStyle = active ? "#00ddff" : "rgba(255,255,255,.7)";
        context.lineWidth = active ? 3 : 1;
        context.strokeRect(x1 * canvas.width / width, y1 * canvas.height / height,
            (x2 - x1) * canvas.width / width, (y2 - y1) * canvas.height / height);
        if (active) {
            context.font = "14px sans-serif";
            const labelWidth = context.measureText(region.name).width + 8;
            const x = Math.max(0, Math.min(canvas.width - labelWidth, x1 * canvas.width / width));
            const y = Math.max(18, y1 * canvas.height / height);
            context.fillStyle = "#002e39"; context.fillRect(x, y - 18, labelWidth, 18);
            context.fillStyle = "#fff"; context.fillText(region.name, x + 4, y - 4);
        }
    };
    const draw = () => {
        if (!width || !base.complete || !base.naturalWidth) return;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(base, 0, 0, canvas.width, canvas.height);
        if (selected) {
            context.fillStyle = "rgba(0,0,0,.5)";
            context.fillRect(0, 0, canvas.width, canvas.height);
            const bits = masks.get(selected.key);
            if (bits) {
                if (shapeKey !== selected.key) {
                    shape = element("canvas"); shape.width = canvas.width; shape.height = canvas.height;
                    const shapeContext = shape.getContext("2d");
                    const pixels = shapeContext.createImageData(canvas.width, canvas.height);
                    for (let i = 0; i < canvas.width * canvas.height; i++) {
                        if (!(bits[i >> 3] & (1 << (i & 7)))) continue;
                        pixels.data[i * 4 + 1] = 210; pixels.data[i * 4 + 2] = 255; pixels.data[i * 4 + 3] = 255;
                    }
                    shapeContext.putImageData(pixels, 0, 0); shapeKey = selected.key;
                }
                const focus = element("canvas"); focus.width = canvas.width; focus.height = canvas.height;
                const focusContext = focus.getContext("2d");
                focusContext.drawImage(base, 0, 0, canvas.width, canvas.height);
                focusContext.globalCompositeOperation = "destination-in";
                focusContext.drawImage(shape, 0, 0);
                context.drawImage(focus, 0, 0);
                context.save(); context.globalAlpha = .2; context.drawImage(shape, 0, 0); context.restore();
            } else {
                const [x1, y1, x2, y2] = selected.bbox;
                const x = x1 * canvas.width / width, y = y1 * canvas.height / height;
                const w = (x2 - x1) * canvas.width / width, h = (y2 - y1) * canvas.height / height;
                context.drawImage(base, x, y, w, h, x, y, w, h);
                context.fillStyle = "rgba(0,210,255,.2)"; context.fillRect(x, y, w, h);
            }
        }
        // Keep every region discoverable while highlighting the active one on top.
        regions.filter(region => region.key !== selected?.key).forEach(region => drawBox(region, false));
        if (selected) drawBox(selected, true);
    };
    base.onload = () => {
        canvas.width = base.naturalWidth; canvas.height = base.naturalHeight;
        [width, height] = row.image_size || [canvas.width, canvas.height];
        canvas.style.cssText = `display:block;margin:auto;width:min(100%,${420 * canvas.width / canvas.height}px);height:auto;cursor:crosshair;touch-action:manipulation`;
        draw();
        for (const [key, descriptor] of Object.entries(row.region_masks || {})) {
            const image = new Image();
            image.onload = () => {
                const buffer = element("canvas"); buffer.width = canvas.width; buffer.height = canvas.height;
                const bufferContext = buffer.getContext("2d");
                bufferContext.drawImage(image, 0, 0, canvas.width, canvas.height);
                const pixels = bufferContext.getImageData(0, 0, canvas.width, canvas.height).data;
                const bits = new Uint8Array(Math.ceil(canvas.width * canvas.height / 8));
                for (let i = 0; i < canvas.width * canvas.height; i++) {
                    if (pixels[i * 4] > 127) bits[i >> 3] |= 1 << (i & 7);
                }
                masks.set(key, bits); if (shapeKey === key) shapeKey = null; draw();
            };
            image.src = imageURL(descriptor);
        }
    };
    base.onerror = () => { status.textContent = "Preview image could not be loaded; run the viewer again."; };
    base.src = imageURL(row.image);
    canvas.addEventListener("pointerdown", event => event.stopPropagation());
    canvas.addEventListener("click", event => {
        event.stopPropagation();
        if (!width) return;
        const rect = canvas.getBoundingClientRect();
        const point = imagePoint(event.clientX, event.clientY, rect, width, height);
        const hits = regionsAt(regions, point.x, point.y, containsMask);
        const signature = hits.map(region => region.key).join("|");
        const repeat = lastClick && lastClick.signature === signature &&
            Math.hypot(event.clientX - lastClick.x, event.clientY - lastClick.y) < 8;
        const next = nextRegion(hits, selected?.option, repeat);
        lastClick = {x: event.clientX, y: event.clientY, signature};
        pick(next?.option ?? 1);
        canvas.focus({preventScroll:true});
    });
    canvas.addEventListener("keydown", event => {
        if (!["ArrowRight", "ArrowLeft", "Escape"].includes(event.key)) return;
        event.preventDefault(); event.stopPropagation(); lastClick = null;
        if (event.key === "Escape" || !regions.length) return pick(1);
        const index = regions.findIndex(region => region.key === selected?.key);
        const offset = event.key === "ArrowRight" ? 1 : -1;
        const next = index < 0 ? (offset > 0 ? 0 : regions.length - 1) : (index + offset + regions.length) % regions.length;
        pick(regions[next].option);
    });
    return {element: wrapper, resetClicks() { lastClick = null; }, select(region, all) {
        selected = region;
        const label = region ? `Selected region: ${region.name}` : all ? "All regions" : "Full image";
        status.textContent = label; canvas.setAttribute("aria-label", label); draw();
    }};
}

app.registerExtension({
    name: "wdv3.results",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["WDV3Preview", "WDV3Inspect"].includes(nodeData.name)) return;
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configure?.apply(this, arguments);
            repairPreviewLimit(this);
            return result;
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            repairPreviewLimit(this);
            return result;
        };
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            previous?.apply(this, arguments);
            if (!this.wdv3Panel) {
                const panel = element("div");
                panel.style.cssText = "overflow:auto;padding:10px;background:#20242c;color:#eee;font:13px sans-serif;box-sizing:border-box;height:100%;border-radius:8px";
                const widget = this.addDOMWidget("wdv3_results", "wdv3_results", panel, { serialize: false });
                const inspect = nodeData.name === "WDV3Inspect";
                widget.computeSize = () => inspect ? [620, 980] : [460, 620];
                this.wdv3Panel = panel;
                this.setSize([Math.max(inspect ? 660 : 500, this.size[0]), Math.max(inspect ? 1100 : 720, this.size[1])]);
            }
            const panel = this.wdv3Panel;
            panel.replaceChildren();
            const rows = message.wdv3_results || [];
            const total = message.wdv3_total?.[0] ?? rows.length;
            panel.append(element("div", `${total} image(s) processed · showing ${rows.length}`));
            if (!rows.length) return;
            const imageChoice = element("select");
            imageChoice.style.cssText = "width:100%;margin:8px 0;padding:6px";
            rows.forEach((row, i) => {
                const detail = row.detail;
                const option = element("option", detail.image_path || `Image ${detail.image_index + 1}`);
                option.value = i;
                imageChoice.append(option);
            });
            const content = element("div");
            panel.append(imageChoice, content);
            const render = () => {
                const row = rows[Number(imageChoice.value)];
                const detail = row.detail;
                content.replaceChildren();
                if (detail.model) {
                    const model=element("div",`${detail.model.model_id} · ${detail.model.device || ""} · ${(detail.image_size || []).join(" × ")}`);
                    model.style.cssText="font-size:11px;color:#bbb;margin:6px 0";
                    model.title=`Weights SHA256: ${detail.model.weights_sha256 || "not recorded"}`;
                    content.append(model);
                }
                const source = element("select");
                source.style.cssText = "width:100%;margin:8px 0;padding:6px";
                const rawStage = detail.stage && detail.stage !== "results";
                const evidence = [[rawStage ? (detail.stage === "regions" ? "Regions" : "Raw full image") : "Merged tags", detail.merged_tags], ["Full image", detail.full_image]];
                const regions = [];
                for (const [name, regions] of Object.entries(detail.detectors || {})) {
                    for (const region of regions) evidence.push([`${name} #${region.region_id} · box ${region.bbox.join(", ")}`, region,
                        {key: `${name}:${region.region_id}`, name: `${name} #${region.region_id}`, bbox: region.bbox, option: evidence.length}]);
                }
                regions.push(...evidence.slice(2).map(entry => entry[2]));
                evidence.forEach(([name], i) => {
                    const option = element("option", name); option.value = i; source.append(option);
                });
                const text = element("textarea");
                text.readOnly = true;
                text.style.cssText = "width:100%;min-height:90px;box-sizing:border-box;resize:vertical";
                const copy = element("button", "Copy tags");
                copy.onclick = () => navigator.clipboard.writeText(text.value);
                const download = element("button", "Download JSON");
                download.style.margin = "6px";
                download.onclick = () => {
                    const url = URL.createObjectURL(new Blob([JSON.stringify(detail, null, 2)], { type: "application/json" }));
                    const link = element("a"); link.href = url;
                    link.download = `wdv3-${detail.image_index + 1}.json`; link.click();
                    setTimeout(() => URL.revokeObjectURL(url), 1000);
                };
                const scores = element("div");
                let imageView;
                let searchDecisions=()=>{};
                const update = () => {
                    const selected = evidence[Number(source.value)];
                    const result = selected[1];
                    imageView?.select(selected[2] || null, Number(source.value) === 0);
                    const format=detail.settings?.options || {};
                    text.value = format.replace_underscores === false ? (result.caption ?? result.taglist ?? "") : (result.taglist || result.caption || "");
                    if(format.escape_parentheses !== false)text.value=text.value.replaceAll("(","\\(").replaceAll(")","\\)");
                    scores.replaceChildren();
                    searchDecisions=()=>{};
                    const decisionRows = Number(source.value) === 0 ? detail.decisions : result.decisions;
                    const records = Array.isArray(decisionRows) ? decisionRows : decisionRows ?
                        Object.entries(decisionRows).flatMap(([category, tags]) => Object.entries(tags).map(([tag, decision]) =>
                            ({tag, category, ...decision, source:selected[2]?.name || "full"}))) :
                        ["general", "character"].flatMap(category => Object.entries(result.raw_scores?.[category] || result[category] || {}).map(([tag,score]) =>
                            ({tag,category,score,source:selected[2]?.name || "full",status:"raw",reason:"raw_evidence"})));
                    if (records.length) {
                        const controls = element("div"), query = element("input"), mode = element("select");
                        query.placeholder = "Search tag, source or reason"; query.style.width = "95%";
                        for (const name of ["all", "accepted", "rejected", "raw"]) {
                            const option=element("option",name);option.value=name;mode.append(option);
                        }
                        const previous=element("button","Previous"), next=element("button","Next"), count=element("span");
                        const table=element("table"); table.style.cssText="width:100%;font-size:11px;overflow-wrap:anywhere";
                        let page=0;
                        const renderDecisions=()=>{
                            const q=query.value.toLowerCase();
                            const filtered=records.filter(r=>(mode.value==="all"||r.status===mode.value)&&
                                `${r.tag} ${r.source} ${r.reason}`.toLowerCase().includes(q));
                            page=Math.min(page,Math.max(0,Math.ceil(filtered.length/50)-1));table.replaceChildren();
                            const header=element("tr");
                            for(const name of ["Tag","Score","Source","Status / reason"])header.append(element("th",name));table.append(header);
                            for(const record of filtered.slice(page*50,(page+1)*50)){
                                const tr=element("tr");
                                for(const value of [record.tag,`${(100*record.score).toFixed(2)}%`,record.source,`${record.status}: ${record.reason}`])tr.append(element("td",value));
                                if(record.region_id!==undefined&&record.region_id!==null)tr.onclick=()=>{
                                    const region=regions.find(r=>r.key===`${record.source}:${record.region_id}`);
                                    if(region){source.value=String(region.option);update();}
                                };
                                table.append(tr);
                            }
                            count.textContent=` ${page+1}/${Math.max(1,Math.ceil(filtered.length/50))} · ${filtered.length} records `;
                            previous.disabled=page===0;next.disabled=(page+1)*50>=filtered.length;
                        };
                        query.oninput=mode.onchange=()=>{page=0;renderDecisions();};
                        previous.onclick=()=>{page--;renderDecisions();}; next.onclick=()=>{page++;renderDecisions();};
                        controls.append(query,mode,previous,count,next);scores.append(controls,table);renderDecisions();
                        searchDecisions=tag=>{query.value=tag;mode.value="all";page=0;renderDecisions();};
                    }
                    const merged = Number(source.value) === 0 && !rawStage;
                    const chooseOrigin = origin => {
                        const region = regions.find(region => region.key === origin.key);
                        if (region) { source.value = String(region.option); imageView?.resetClicks(); update(); }
                    };
                    if (merged) {
                        const additions = ["character", "general"].flatMap(category =>
                            Object.keys(result[category] || {}).map(tag => ({tag, origin: mergedTagOrigin(detail, category, tag)})))
                            .filter(entry => entry.origin.added);
                        const added = element("details"); added.open = true;
                        added.append(element("summary", `Added from regions (${additions.length})`));
                        if (!additions.length) added.append(element("div", "No new region tags in the merged result."));
                        for (const {tag, origin} of additions) {
                            const button = element("button", `${tag} · ${origin.label}`);
                            button.style.cssText = "margin:3px;padding:4px 6px;color:#9deaff;border:1px solid #397586;border-radius:5px";
                            button.title = "Show the source region";
                            button.onclick = () => chooseOrigin(origin);
                            added.append(button);
                        }
                        scores.append(added);
                        const removed=["character","general"].flatMap(category=>Object.keys(detail.full_image?.[category]||{})
                            .filter(tag=>!Object.hasOwn(result[category]||{},tag)).map(tag=>({tag,category})));
                        const missing=element("details");missing.append(element("summary",`Removed from full image (${removed.length})`));
                        for(const {tag,category}of removed){
                            const reason=detail.decisions?.find(d=>d.tag===tag&&d.category===category&&d.source==="full"&&d.status==="rejected")?.reason||"removed";
                            const button=element("button",`${tag} · ${reason}`);button.style.cssText="margin:3px;color:#ffb8b8";
                            button.onclick=()=>searchDecisions(tag);missing.append(button);
                        }
                        scores.append(missing);
                    }
                    if (result.segmentation_score !== undefined) scores.append(element("div", `SAM confidence: ${(100 * result.segmentation_score).toFixed(1)}%`));
                    for (const category of ["ratings", "character", "general"]) {
                        const entries = Object.entries(result[category] || {}).sort((a, b) => b[1] - a[1]);
                        if (!entries.length) continue;
                        const group = element("details"); group.open = category !== "general";
                        group.append(element("summary", `${category} (${entries.length})`));
                        const table = element("table"); table.style.cssText = "width:100%;border-collapse:collapse";
                        for (const [tag, score] of entries.slice(0, 50)) {
                            const tr = element("tr");
                            tr.append(element("td", tag), element("td", `${(100 * score).toFixed(2)}%`));
                            if (merged && category !== "ratings") {
                                const origin = mergedTagOrigin(detail, category, tag);
                                const cell = element("td");
                                if (origin.key) {
                                    const button = element("button", `${origin.added ? "+ " : ""}${origin.label}`);
                                    button.onclick = () => chooseOrigin(origin); cell.append(button);
                                } else cell.textContent = origin.label;
                                tr.append(cell);
                            }
                            table.append(tr);
                        }
                        if(entries.length>50)group.append(element("div","Top 50 shown; use the searchable table for all scores."));
                        group.append(table); scores.append(group);
                    }
                };
                source.onchange = () => { imageView?.resetClicks(); update(); };
                if (row.image) {
                    imageView = regionCanvas(row, regions, option => { source.value = String(option); update(); });
                    content.append(imageView.element);
                }
                content.append(source, text, copy, download, scores); update();
            };
            imageChoice.onchange = render; render();
            app.graph.setDirtyCanvas(true, true);
        };
    }
});
