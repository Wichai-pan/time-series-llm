import pptxgen from "pptxgenjs";

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Huatai Pan";
pptx.subject = "Advisor update on SDForger PAMAP2 verification experiments";
pptx.title = "SDForger 在 PAMAP2 上的会后验证实验";
pptx.company = "Aalto ELEC-E7633 Project Course";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "zh-CN"
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const C = {
  ink: "0F172A",
  muted: "64748B",
  line: "CBD5E1",
  panel: "F8FAFC",
  teal: "0F766E",
  tealBg: "ECFDF5",
  amber: "F59E0B",
  red: "DC2626",
  blue: "2563EB",
  white: "FFFFFF"
};

const W = 13.333;
const H = 7.5;
const marginX = 0.52;

function addTopRule(slide) {
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.12,
    fill: { color: C.teal },
    line: { color: C.teal }
  });
}

function addFooter(slide, page) {
  slide.addText("SDForger / PAMAP2 · Advisor update · 2026-06-08", {
    x: marginX,
    y: 7.12,
    w: 8.5,
    h: 0.18,
    fontFace: "Aptos",
    fontSize: 6.8,
    color: C.muted,
    margin: 0
  });
  slide.addText(String(page), {
    x: 12.3,
    y: 7.07,
    w: 0.48,
    h: 0.24,
    fontFace: "Aptos",
    fontSize: 7.5,
    bold: true,
    align: "center",
    color: C.muted,
    margin: 0
  });
}

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: marginX,
    y: 0.38,
    w: 11.8,
    h: 0.42,
    fontFace: "Aptos Display",
    fontSize: 24,
    bold: true,
    color: C.ink,
    margin: 0
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: marginX,
      y: 0.86,
      w: 11.8,
      h: 0.24,
      fontFace: "Aptos",
      fontSize: 9,
      color: C.muted,
      margin: 0
    });
  }
}

function addPanel(slide, x, y, w, h, title, bodyLines, accent = C.teal) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    fill: { color: C.panel },
    line: { color: C.line, width: 0.8 }
  });
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 0.07,
    h,
    fill: { color: accent },
    line: { color: accent }
  });
  slide.addText(title, {
    x: x + 0.22,
    y: y + 0.16,
    w: w - 0.42,
    h: 0.26,
    fontFace: "Aptos Display",
    fontSize: 12.2,
    bold: true,
    color: C.ink,
    margin: 0
  });
  slide.addText(bodyLines.map((line) => ({ text: line, options: { bullet: { type: "ul" } } })), {
    x: x + 0.28,
    y: y + 0.54,
    w: w - 0.52,
    h: h - 0.7,
    fontFace: "Aptos",
    fontSize: 9.3,
    color: C.ink,
    breakLine: false,
    fit: "shrink",
    paraSpaceAfterPt: 5,
    margin: 0
  });
}

function tableCell(text, opts = {}) {
  return {
    text,
    options: {
      fontFace: "Aptos",
      fontSize: opts.fontSize ?? 7.2,
      bold: opts.bold ?? false,
      color: opts.color ?? C.ink,
      fill: opts.fill ? { color: opts.fill } : undefined,
      align: opts.align ?? "left",
      valign: "mid",
      margin: opts.margin ?? 0.06,
      fit: "shrink"
    }
  };
}

function addTakeaway(slide, text, y) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: marginX,
    y,
    w: 12.25,
    h: 0.48,
    rectRadius: 0.04,
    fill: { color: C.tealBg },
    line: { color: "A7F3D0", width: 0.7 }
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: marginX,
    y,
    w: 0.08,
    h: 0.48,
    fill: { color: C.teal },
    line: { color: C.teal }
  });
  slide.addText(text, {
    x: marginX + 0.22,
    y: y + 0.11,
    w: 11.75,
    h: 0.24,
    fontFace: "Aptos",
    fontSize: 10.2,
    bold: true,
    color: "134E4A",
    margin: 0
  });
}

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  addTopRule(slide);
  addTitle(slide, "SDForger 在 PAMAP2 上的会后验证实验", "目标：确认 unified activity-conditioned generation 的不稳定来源，并确定下一步方法方向");

  addPanel(
    slide,
    marginX,
    1.28,
    5.92,
    1.62,
    "当前任务",
    [
      "Activity-conditioned sensor generation",
      "数据集：PAMAP2；动作：walking / running",
      "通道：hand_acc16_x",
      "基础流程：SDForger + FICA latent + LLM generation"
    ],
    C.blue
  );
  addPanel(
    slide,
    6.9,
    1.28,
    5.88,
    1.62,
    "主要问题",
    [
      "为什么 raw unified generation 会不稳定？",
      "是 scale mismatch、数据量不足，还是 generated latent 本身越界？",
      "当前结果是 diagnostic verification，不是最终 HAR augmentation claim"
    ],
    C.amber
  );

  const rows = [
    [
      tableCell("实验", { bold: true, fill: "E2E8F0", fontSize: 7.6 }),
      tableCell("设置", { bold: true, fill: "E2E8F0", fontSize: 7.6 }),
      tableCell("目的", { bold: true, fill: "E2E8F0", fontSize: 7.6 }),
      tableCell("当前发现", { bold: true, fill: "E2E8F0", fontSize: 7.6 })
    ],
    [
      tableCell("归一化", { bold: true }),
      tableCell("window / joint / global / activity-level z-score"),
      tableCell("检查 scale mismatch 是否导致数值爆炸"),
      tableCell("方向合理，但单靠 normalization 没有稳定 unified generation")
    ],
    [
      tableCell("Multi-subject", { bold: true }),
      tableCell("train subjects 101 / 102 / 105"),
      tableCell("检查 subject101 数据量太少是不是主因"),
      tableCell("增加 subject 后，raw unified 仍然不稳定")
    ],
    [
      tableCell("Unseen subject", { bold: true }),
      tableCell("reference subjects 106 / 108"),
      tableCell("检查 synthetic windows 是否只贴近训练 subject"),
      tableCell("clip 在 held-out reference 上仍稳定，但还不是 HAR utility 结论")
    ],
    [
      tableCell("Latent validity", { bold: true }),
      tableCell("post-hoc clip / strict reject / soft repair"),
      tableCell("在 decode 前控制无效 latent 数值"),
      tableCell("simple clip 是最强 diagnostic baseline；soft repair 更适合作为方法候选")
    ]
  ];
  slide.addTable(rows, {
    x: marginX,
    y: 3.25,
    w: 12.25,
    h: 2.5,
    colW: [1.45, 3.15, 3.35, 4.3],
    rowH: [0.38, 0.52, 0.52, 0.52, 0.56],
    border: { type: "solid", color: C.line, width: 0.6 },
    valign: "mid"
  });
  addTakeaway(slide, "目前看，问题不只是数据量或归一化；unified generation 需要 latent validity control。", 6.08);
  addFooter(slide, 1);
  slide.addNotes(`这页先讲这次做了什么，不要一开始陷入指标。

我们沿着老师上次反馈做了四组检查：归一化、多 subject、unseen subject、以及不同的 clip/validity control。

重点不是说我们已经有最终方法，而是把问题定位得更清楚：raw unified activity-conditioned generation 容易产生无效 latent，经过 FICA decode 后会变成极端异常波形。`);
}

// Slide 2
{
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  addTopRule(slide);
  addTitle(slide, "主要结果：clip 能明显稳定 unified generation", "TSGBench-style diagnostic metrics；这里先用 DTW 说明核心现象，DTW 越低越好");

  const rows = [
    [
      tableCell("设置", { bold: true, fill: "E2E8F0", fontSize: 7.8 }),
      tableCell("参考真实数据", { bold: true, fill: "E2E8F0", fontSize: 7.8 }),
      tableCell("Walking DTW ↓", { bold: true, fill: "E2E8F0", fontSize: 7.8, align: "right" }),
      tableCell("Running DTW ↓", { bold: true, fill: "E2E8F0", fontSize: 7.8, align: "right" }),
      tableCell("解释", { bold: true, fill: "E2E8F0", fontSize: 7.8 })
    ],
    [
      tableCell("raw unified", { bold: true, color: C.red }),
      tableCell("train 101/102/105"),
      tableCell("9920.9", { align: "right", color: C.red, bold: true }),
      tableCell("7943.9", { align: "right", color: C.red, bold: true }),
      tableCell("多 subject 后仍然生成不稳定")
    ],
    [
      tableCell("clip p05-p95", { bold: true, color: C.teal }),
      tableCell("train 101/102/105"),
      tableCell("166.1", { align: "right", color: C.teal, bold: true }),
      tableCell("147.2", { align: "right", color: C.teal, bold: true }),
      tableCell("简单 latent clip 把 DTW 拉回合理范围")
    ],
    [
      tableCell("raw unified", { bold: true, color: C.red }),
      tableCell("held-out 106/108"),
      tableCell("11160.4", { align: "right", color: C.red, bold: true }),
      tableCell("9567.4", { align: "right", color: C.red, bold: true }),
      tableCell("面对 unseen-subject reference 时仍失败")
    ],
    [
      tableCell("clip p05-p95", { bold: true, color: C.teal }),
      tableCell("held-out 106/108"),
      tableCell("151.1", { align: "right", color: C.teal, bold: true }),
      tableCell("146.7", { align: "right", color: C.teal, bold: true }),
      tableCell("clip 在 unseen-subject reference 上仍稳定")
    ],
    [
      tableCell("soft repair", { bold: true, color: C.blue }),
      tableCell("held-out 106/108"),
      tableCell("140.0", { align: "right", color: C.blue, bold: true }),
      tableCell("157.2", { align: "right", color: C.blue, bold: true }),
      tableCell("更可解释，但目前还没有明显优于 simple clip")
    ]
  ];
  slide.addTable(rows, {
    x: marginX,
    y: 1.28,
    w: 12.25,
    h: 3.05,
    colW: [1.75, 2.1, 1.55, 1.55, 5.3],
    rowH: [0.42, 0.48, 0.48, 0.48, 0.48, 0.5],
    border: { type: "solid", color: C.line, width: 0.6 },
    valign: "mid"
  });

  addPanel(
    slide,
    marginX,
    4.62,
    5.95,
    1.23,
    "可以稳妥汇报的结论",
    [
      "clip_p05_p95 是 strong diagnostic baseline",
      "结果支持把 latent validity control 作为下一步方法方向"
    ],
    C.teal
  );
  addPanel(
    slide,
    6.9,
    4.62,
    5.88,
    1.23,
    "下一步讨论",
    [
      "从 post-hoc clip 推进到 generation-time validity + resampling",
      "报告 clean / repaired / rejected / malformed counts",
      "生成质量稳定后，再补 held-out HAR utility"
    ],
    C.blue
  );

  addTakeaway(slide, "下一步关键决策：是否把 simple clipping 扩展成更正式的 soft validity control，并用 held-out HAR utility 评估。", 6.18);
  addFooter(slide, 2);
  slide.addNotes(`这页只讲 DTW，是为了避免表格太复杂。完整指标还有 MDD、ACD、SD、KD、ED、DTW，但是汇报时先用 DTW 说明核心现象最清楚。

raw unified 的 DTW 非常大，说明生成曲线和真实 walking/running window 差很远。多 subject 后仍然很大，说明不是简单因为 subject101 数据少。

clip p05-p95 的含义是：LLM 生成 FICA latent 后，把每一维限制在训练 latent 的 5% 到 95% 分位数范围内，再 decode 回 sensor window。它不是最终方法，但能明显压住极端 latent，说明 failure mode 很可能发生在 generated latent validity 上。

unseen reference 是用 106/108 的真实数据做参考，不是重新训练。这个结果不能说已经证明泛化，但能说明 clip 后的 synthetic windows 不是只贴近训练 subject。

soft repair 比 clip 更像一个方法：先判断 latent 是否超出合理范围，能修就修，太离谱就拒绝。但现在它没有全面超过 simple clip，所以后续需要配合 resampling，补回样本数，再做 HAR utility。`);
}

await pptx.writeFile({ fileName: "decks/2026-06-08-advisor-postmeeting-update.pptx" });
