const track1Data = [
  { label: "Baseline A", reference: 51.5667, reconstructed: 0.000251 },
  { label: "Enhanced", reference: 54.3868, reconstructed: 0.0 },
  { label: "Geo RF", reference: 53.8357, reconstructed: 0.021675 },
  { label: "Geo FR", reference: 54.4261, reconstructed: 0.150146 },
  { label: "Impl A", reference: 54.6734, reconstructed: 0.811525 },
  { label: "Impl B", reference: 53.5699, reconstructed: 0.019433 },
  { label: "Pillar A", reference: 54.7708, reconstructed: 0.018513 },
  { label: "Pillar B", reference: 52.7378, reconstructed: 2.192242 }
];

const track2Data = [
  { label: "Raw/basic", value: 0.497898, tone: "highlight" },
  { label: "Stage0 baseline", value: 0.002823, tone: "reference" },
  { label: "Stage0 enhanced", value: 0.001048, tone: "reference" },
  { label: "Stage1 baseline", value: 0.00072, tone: "reference" },
  { label: "Stage1 enhanced", value: 0.00029, tone: "reference" }
];

const oracleData = [
  { label: "Stage1 native", value: 0.8064, tone: "reference" },
  { label: "Stage1 oracle ROI", value: 0.4177, tone: "alt" },
  { label: "Stage2 native", value: 0.8687, tone: "reference" },
  { label: "Stage2 oracle ROI", value: 0.4492, tone: "alt" }
];

function observeSections() {
  const items = Array.from(document.querySelectorAll("[data-reveal]"));
  if (!("IntersectionObserver" in window) || items.length === 0) {
    items.forEach((item) => item.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.14 }
  );

  items.forEach((item) => observer.observe(item));
}

function svgNode(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function addText(svg, attrs, text) {
  const node = svgNode("text", attrs);
  node.textContent = text;
  svg.appendChild(node);
  return node;
}

function renderTrack1Chart() {
  const svg = document.getElementById("track1-chart");
  if (!svg) return;
  svg.replaceChildren();

  const width = 760;
  const height = 430;
  const margin = { top: 28, right: 28, bottom: 24, left: 120 };
  const plotWidth = width - margin.left - margin.right;
  const rowHeight = 40;
  const barHeight = 10;
  const maxValue = 56;

  track1Data.forEach((entry, index) => {
    const rowTop = margin.top + index * rowHeight;
    const yLabel = rowTop + 16;
    const yReference = rowTop + 20;
    const yRecon = rowTop + 34;
    const refWidth = (entry.reference / maxValue) * plotWidth;
    const reconWidth = (entry.reconstructed / maxValue) * plotWidth;

    addText(svg, { x: 0, y: yLabel, class: "chart-label" }, entry.label);

    const refBar = svgNode("rect", {
      x: margin.left,
      y: yReference,
      width: refWidth,
      height: barHeight,
      rx: 5,
      class: "bar-reference"
    });
    svg.appendChild(refBar);

    const reconBar = svgNode("rect", {
      x: margin.left,
      y: yRecon,
      width: Math.max(reconWidth, entry.reconstructed > 0 ? 2 : 0),
      height: barHeight,
      rx: 5,
      class: entry.label === "Pillar B" ? "bar-highlight" : "bar-alt"
    });
    svg.appendChild(reconBar);

    addText(svg, { x: margin.left + refWidth + 8, y: yReference + 9, class: "chart-value" }, entry.reference.toFixed(2));
    addText(
      svg,
      { x: margin.left + Math.max(reconWidth, entry.reconstructed > 0 ? 2 : 0) + 8, y: yRecon + 9, class: "chart-value" },
      entry.reconstructed.toFixed(2)
    );
  });

  addText(svg, { x: margin.left, y: 16, class: "legend-label" }, "Reference");
  svg.appendChild(svgNode("rect", { x: margin.left - 18, y: 8, width: 12, height: 12, rx: 3, class: "bar-reference" }));
  addText(svg, { x: margin.left + 110, y: 16, class: "legend-label" }, "Reconstructed");
  svg.appendChild(svgNode("rect", { x: margin.left + 90, y: 8, width: 12, height: 12, rx: 3, class: "bar-alt" }));
}

function renderHorizontalChart(svgId, data, options) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  svg.replaceChildren();

  const width = options.width;
  const height = options.height;
  const margin = options.margin;
  const plotWidth = width - margin.left - margin.right;
  const rowHeight = options.rowHeight;
  const maxValue = options.maxValue;

  if (options.gridTicks) {
    options.gridTicks.forEach((tick) => {
      const x = margin.left + (tick / maxValue) * plotWidth;
      svg.appendChild(svgNode("line", { x1: x, y1: margin.top - 4, x2: x, y2: height - margin.bottom, class: "axis-line" }));
      addText(svg, { x, y: height - 6, class: "grid-label", "text-anchor": "middle" }, tick.toFixed(options.tickPrecision ?? 2));
    });
  }

  data.forEach((entry, index) => {
    const rowTop = margin.top + index * rowHeight;
    const barY = rowTop + 6;
    const barWidth = (entry.value / maxValue) * plotWidth;

    addText(svg, { x: 0, y: rowTop + 18, class: "chart-label" }, entry.label);
    svg.appendChild(
      svgNode("rect", {
        x: margin.left,
        y: barY,
        width: barWidth,
        height: 18,
        rx: 8,
        class: entry.tone === "highlight" ? "bar-highlight" : entry.tone === "alt" ? "bar-alt" : "bar-reference"
      })
    );
    addText(svg, { x: margin.left + barWidth + 8, y: barY + 14, class: "chart-value" }, entry.value.toFixed(options.valuePrecision ?? 3));
  });
}

document.addEventListener("DOMContentLoaded", () => {
  observeSections();
  renderTrack1Chart();
  renderHorizontalChart("track2-chart", track2Data, {
    width: 760,
    height: 340,
    margin: { top: 20, right: 40, bottom: 28, left: 160 },
    rowHeight: 50,
    maxValue: 0.52,
    gridTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    tickPrecision: 1,
    valuePrecision: 4
  });
  renderHorizontalChart("oracle-chart", oracleData, {
    width: 760,
    height: 280,
    margin: { top: 20, right: 40, bottom: 28, left: 170 },
    rowHeight: 52,
    maxValue: 0.92,
    gridTicks: [0, 0.2, 0.4, 0.6, 0.8],
    tickPrecision: 1,
    valuePrecision: 4
  });
});
