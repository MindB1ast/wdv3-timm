export function imagePoint(clientX, clientY, rect, width, height) {
    return {x: (clientX - rect.left) * width / rect.width,
            y: (clientY - rect.top) * height / rect.height};
}

export function regionsAt(regions, x, y, maskContains = () => true) {
    return regions.filter(region => {
        const [x1, y1, x2, y2] = region.bbox;
        return x >= x1 && x < x2 && y >= y1 && y < y2 && maskContains(region, x, y);
    }).sort((a, b) => {
        const area = region => (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]);
        return area(a) - area(b) || a.option - b.option;
    });
}

export function nextRegion(hits, selected, repeat) {
    if (!hits.length) return null;
    const index = repeat ? hits.findIndex(region => region.option === selected) : -1;
    return hits[(index + 1) % hits.length];
}
