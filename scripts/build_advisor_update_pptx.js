const pptxgen = require("/Users/huataipan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.subject = "Advisor update: Activity-Conditioned SDForger on PAMAP2";
pptx.title = "Activity-Conditioned SDForger on PAMAP2";
pptx.company = "Aalto Project Course";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });

const assetDir = "slides/decks/assets/2026-05-26-advisor-sdforger-pamap2-update";
const colors = {
  ink: "0F172A",
  muted: "64748B",
  border: "D8DEE9",
  soft: "F8FAFC",
  greenBg: "ECFDF5",
  green: "0F766E",
  greenText: "134E4A",
  tableHead: "F1F5F9",
};

function addTitle(slide, title) {
  slide.addText(title, {
    x: 0.55,
    y: 0.32,
    w: 12.2,
    h: 0.45,
    fontFace: "Aptos Display",
    fontSize: 27,
    bold: true,
    color: colors.ink,
    margin: 0,
    breakLine: false,
    fit: "shrink",
  });
}

function addTakeaway(slide, text, y = 6.75) {
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.55,
    y,
    w: 12.2,
    h: 0.48,
    fill: { color: colors.greenBg },
    line: { color: colors.greenBg },
    radius: 0.08,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.55,
    y,
    w: 0.07,
    h: 0.48,
    fill: { color: colors.green },
    line: { color: colors.green },
  });
  slide.addText(text, {
    x: 0.74,
    y: y + 0.09,
    w: 11.85,
    h: 0.3,
    fontSize: 14,
    bold: true,
    color: colors.greenText,
    margin: 0,
    fit: "shrink",
  });
}

function addCard(slide, x, y, w, h, title, bullets) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    fill: { color: "FFFFFF" },
    line: { color: colors.border, width: 1 },
  });
  slide.addText(title, {
    x: x + 0.18,
    y: y + 0.17,
    w: w - 0.36,
    h: 0.25,
    fontSize: 14,
    bold: true,
    color: colors.ink,
    margin: 0,
  });
  slide.addText(
    bullets.map((b) => ({ text: b, options: { bullet: { type: "bullet" } } })),
    {
      x: x + 0.28,
      y: y + 0.58,
      w: w - 0.46,
      h: h - 0.72,
      fontSize: 11.5,
      color: colors.ink,
      breakLine: false,
      fit: "shrink",
      valign: "top",
      paraSpaceAfterPt: 5,
      margin: 0,
    }
  );
}

function addTable(slide, rows, x, y, w, h, colW) {
  const rowH = h / rows.length;
  rows.forEach((row, r) => {
    let cx = x;
    row.forEach((cell, c) => {
      const cw = colW[c] * w;
      slide.addShape(pptx.ShapeType.rect, {
        x: cx,
        y: y + r * rowH,
        w: cw,
        h: rowH,
        fill: { color: r === 0 ? colors.tableHead : "FFFFFF" },
        line: { color: colors.border, width: 0.75 },
      });
      slide.addText(cell, {
        x: cx + 0.08,
        y: y + r * rowH + 0.08,
        w: cw - 0.16,
        h: rowH - 0.12,
        fontSize: r === 0 ? 10.5 : 10.2,
        bold: r === 0,
        color: colors.ink,
        margin: 0,
        align: c > 0 ? "right" : "left",
        fit: "shrink",
      });
      cx += cw;
    });
  });
}

function addCaption(slide, text, x, y, w) {
  slide.addText(text, {
    x,
    y,
    w,
    h: 0.18,
    fontSize: 8.5,
    color: colors.muted,
    margin: 0,
    fit: "shrink",
  });
}

function addNoteBox(slide, text, x, y, w, h) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.05,
    fill: { color: colors.soft },
    line: { color: colors.border, width: 1 },
  });
  slide.addText(text, {
    x: x + 0.16,
    y: y + 0.14,
    w: w - 0.32,
    h: h - 0.26,
    fontSize: 11,
    color: colors.ink,
    margin: 0,
    fit: "shrink",
    valign: "mid",
  });
}

// Slide 1
{
  const slide = pptx.addSlide();
  addTitle(slide, "Activity-Conditioned SDForger on PAMAP2");
  slide.addText("Advisor update · 3-slide progress summary", {
    x: 0.58,
    y: 0.85,
    w: 5.5,
    h: 0.18,
    fontSize: 9.5,
    bold: true,
    color: colors.muted,
    margin: 0,
  });
  addCard(slide, 0.6, 1.2, 5.85, 2.0, "Previous recovered setup", [
    "PAMAP2 subject101",
    "Mixed non-zero activities",
    "Single channel: hand_acc16_x",
    "No activity label in processed parquet",
    "Unconditioned SDForger-style generation",
  ]);
  addCard(slide, 6.85, 1.2, 5.85, 2.0, "Current controlled setup", [
    "PAMAP2 subject101",
    "Walking / running only",
    "Single channel: hand_acc16_x",
    "Activity text as condition",
    "Baseline + conditional generation",
  ]);
  slide.addImage({
    path: `${assetDir}/running-periodicity.png`,
    x: 1.45,
    y: 3.45,
    w: 10.5,
    h: 2.35,
    sizing: { type: "contain", x: 1.45, y: 3.45, w: 10.5, h: 2.35 },
  });
  addCaption(slide, "Running was selected as a controlled periodic HAR signal; walking is the paired activity.", 1.45, 5.88, 10.5);
  addTakeaway(slide, "We moved from recovering a mixed-activity baseline to a controlled HAR generation task.");
}

// Slide 2
{
  const slide = pptx.addSlide();
  addTitle(slide, "Activity conditioning works, but the unified generator is unstable");
  addNoteBox(
    slide,
    "Condition: data is walking/running\n+ FICA embedding text\n→ LLM fine-tuning / generation\n→ decoded synthetic sensor window",
    0.65,
    1.14,
    4.55,
    1.35
  );
  addTable(
    slide,
    [
      ["Diagnostic", "Raw unified"],
      ["Label controllability", "0.8212"],
      ["Walking abs max", "3761.8"],
      ["Running abs max", "1572.1"],
      ["Synthetic-only HAR", "0.4955"],
    ],
    0.65,
    2.72,
    4.55,
    2.2,
    [0.64, 0.36]
  );
  slide.addImage({
    path: `${assetDir}/running-unified-raw-acf-psd.png`,
    x: 5.55,
    y: 1.25,
    w: 7.1,
    h: 4.0,
    sizing: { type: "contain", x: 5.55, y: 1.25, w: 7.1, h: 4.0 },
  });
  addCaption(slide, "Unified label-conditioned generation before latent constraint.", 5.55, 5.32, 7.1);
  addTakeaway(slide, "The model learns activity-label signal, but unconstrained generated latents can leave the valid sensor range.");
}

// Slide 3
{
  const slide = pptx.addSlide();
  addTitle(slide, "Latent validity constraint improves stability");
  addTable(
    slide,
    [
      ["Metric", "Raw unified", "clip_p05_p95"],
      ["Walking abs max", "3761.8", "3.24"],
      ["Running abs max", "1572.1", "2.65"],
      ["Label controllability", "0.8212", "0.9868"],
      ["Synthetic-only HAR", "0.4955", "0.6937"],
    ],
    0.65,
    1.15,
    5.15,
    2.38,
    [0.52, 0.24, 0.24]
  );
  addNoteBox(
    slide,
    "Next method step:\nGenerate candidate latent → check validity → accept/reject → decode.",
    0.65,
    3.9,
    5.15,
    0.9
  );
  slide.addImage({
    path: `${assetDir}/running-constrained-acf-psd.png`,
    x: 6.05,
    y: 1.25,
    w: 6.65,
    h: 4.0,
    sizing: { type: "contain", x: 6.05, y: 1.25, w: 6.65, h: 4.0 },
  });
  addCaption(slide, "After percentile latent constraint, running recovers reasonable value scale and rhythm diagnostics.", 6.05, 5.32, 6.65);
  addTakeaway(slide, "The current direction is controlled conditional generation with latent validity checks.");
}

pptx.writeFile({ fileName: "slides/decks/2026-05-26-advisor-sdforger-pamap2-update.pptx" });
