export function mergedTagOrigin(detail, category, tag) {
    const provenance = detail.merged_tags?.provenance?.[tag];
    const source = provenance?.region_source ?? provenance?.source;
    const region = Object.entries(detail.detectors || {}).flatMap(([name, rows]) =>
        rows.map(row => ({key: `${name}:${row.region_id}`, label: `${name} #${row.region_id}`})))
        .find(region => region.key === source);
    return {
        key: region?.key,
        label: region ? `${region.label}${provenance.source === "manual_rule" ? " · rule" : ""}` :
            source === "full" ? "Full image" : source === "manual_rule" ? "Manual rule" : "Unknown source",
        added: Boolean(region && !Object.hasOwn(detail.full_image?.[category] || {}, tag)),
    };
}
