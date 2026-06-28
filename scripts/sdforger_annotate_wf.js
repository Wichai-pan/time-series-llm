export const meta = {
  name: 'sdforger-code-annotate',
  description: 'Annotate official SDForger code with Chinese paper-mapping comments + return verified map',
  phases: [
    { title: 'Annotate', detail: 'parallel: one agent per core file, add Chinese paper-mapping comments' },
    { title: 'Synthesize', detail: 'verified code to paper mapping table' },
  ],
}

const DIR = '/Users/huataipan/Wichai/CODAS/Aalto/ELEC-E7633 - Project Course/reference/sdforger-official'

const PAPER = [
  'SDForger (Forging Time Series with Language, arXiv:2505.17103v2) method, for code-mapping:',
  '6-step pipeline: (1) periodicity-aware segmentation -> (2) ICA/FPC embedding -> (3) textual encoding -> (4) LLM finetune + inference -> (5) textual decoding -> (6) ICA/FPC decoding back to time series.',
  'Sec3.1 time series to tabular: project each window onto k basis functions; embedding coeff e_ij = inner_product(X_i, b_j). Embedding table E with shape I x K, K = sum of k_c. Two basis options: FPC (Functional Principal Components, covariance-based, ordered by variance, parsimonious) and FastICA (statistically independent components, unordered, info spread evenly, more robust in generation). Cost depends on number of instances I and components k, NOT on sequence length L.',
  'A.1 segmentation (single series): estimate dominant period P via ACF (significant peaks excluding lag 0, ranked by autocorrelation, pick highest with P < L/2). Window step s = max(1, floor((L0-L)/(I-1))) adjusted to nearest multiple of P. About I=15 instances is enough.',
  'A.2 choice of k: smallest k explaining a target variance percentage, or manual. K>25 leads to unstable finetuning and more discarded samples.',
  'Sec3.2.1 textual encoding: fill-in-the-middle template. Roughly: "Input: value_pi(k) is [blank], ... [sep] Target: e_i,pi(k) [answer] ...". pi is a RANDOM PERMUTATION of the K features applied per instance, to remove positional/order bias.',
  'Sec3.2.2 LLM finetune: GPT-2, Adam lr 8e-5, batch 32, up to 200 epochs, EARLY STOPPING (val loss every 5 steps, patience 5, 20 percent as val). Inference: prompt with blanks, MULTINOMIAL / temperature sampling for diversity; model autoregressively fills the [answer] coefficients.',
  'A.3 in-generation filtering: discard (1) instances with missing values, (2) duplicated instances, (3) diverging instances. Diverging = squared L2-norm of embedding vector outside [q1 - 3*IQR, q3 + 3*IQR] of original norms (per channel). For FPC the L2-norm of the curve approximates the Euclidean norm of coefficients.',
  'A.4 stopping criterion: diversity score D = (number of unique norms rounded to 4 decimals) / (number of valid instances). Stop when max over channels of D < lambda_stop OR count > I_max.',
  'Sec3.3 decoding: reconstruct x_i = sum over j of (e_ij * b_j) (linear combination of generated coefficients and basis).',
  'Eval metrics (App B): feature-based MDD (marginal distribution diff), ACD (autocorrelation diff), SD (skewness diff), KD (kurtosis diff); distance-based ED (euclidean), DTW (dynamic time warping), SHR (shapelet reconstruction).',
].join('\n')

const RULES = [
  'ANNOTATION RULES (strict):',
  '- Read the assigned file fully first. Then ADD Chinese comment lines that map code to the paper. Do NOT change, delete, or reorder any existing code or existing comments. ONLY insert new comment lines.',
  '- EVERY comment you add MUST start with the exact prefix: # paper-note-added: (Chinese text after it). This makes it unambiguously a later-added annotation, not original code. (Use this exact ASCII prefix so it is greppable.)',
  '- Add: (a) a block annotation above each major class/function saying which paper step/section it implements and what it does in one line of Chinese; (b) inline annotations on the few KEY lines (the ICA/FPC call, the inner-product embedding, the text template construction, the random permutation, the sampling call, the L2-norm IQR filter, the linear decode, the ACF period estimation) explaining the correspondence in Chinese.',
  '- Keep annotations concise and accurate. If a code section is framework plumbing with no clear paper mapping, say so briefly in Chinese rather than forcing a mapping.',
  '- Use the Edit tool to insert the comments. Preserve indentation so the file still parses as valid Python/YAML.',
].join('\n')

const MAP_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    annotations_added: { type: 'integer' },
    mapping: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          code_symbol: { type: 'string' },
          paper_section: { type: 'string' },
          what_it_does: { type: 'string' },
        },
        required: ['code_symbol', 'paper_section', 'what_it_does'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['file', 'annotations_added', 'mapping', 'notes'],
}

const FILES = [
  { f: 'utils.py', hint: 'expect: periodicity-aware segmentation (ACF period), FPC/FastICA embedding (inner-product coeffs), decoding (linear reconstruction), possibly metrics. Paper Sec3.1, Sec3.3, A.1, A.2, App B.' },
  { f: 'generate.py', hint: 'expect: textual encoding (fill-in-the-middle template + random permutation), inference sampling, retrieve-embedding-from-text, in-generation filtering (L2-norm IQR), stopping criterion. Paper Sec3.2.1, A.3, A.4.' },
  { f: 'trainer.py', hint: 'expect: LLM fine-tuning loop (GPT-2, Adam, early stopping). Paper Sec3.2.2.' },
  { f: 'data_objects.py and task.py and time_series.yaml', hint: 'expect: data structures, task definition wiring SDForger into fms-dgt, config (embedding dim k, LLM, hyperparams). Annotate all three.' },
]

phase('Annotate')
const results = (await parallel(FILES.map(function (item) {
  return function () {
    const prompt = 'You are mapping the official SDForger code to its paper, annotating in Chinese. Files live under: ' + DIR +
      '\nYour assigned file(s): ' + item.f +
      '\nLikely content: ' + item.hint +
      '\n\n' + PAPER +
      '\n\n' + RULES +
      '\n\nRead the assigned file(s) and add the Chinese paper-mapping annotations as specified. Then return the structured mapping of code symbols to paper sections. If assigned multiple files, set "file" to a comma-joined list and cover all in "mapping".'
    return agent(prompt, { label: 'annotate:' + item.f.split(' ')[0], phase: 'Annotate', schema: MAP_SCHEMA })
  }
}))).filter(Boolean)

log('Annotated ' + results.length + '/' + FILES.length + ' file groups')

phase('Synthesize')
const synthPrompt = 'You have the per-file code-to-paper mappings for the official SDForger code (just annotated with Chinese comments). Produce a concise Chinese Markdown briefing for a student preparing a faithful 15-20 min presentation:\n' +
  '1. Code-to-paper master table: one row per major code symbol: file, function/class, corresponding paper section, one-line role. Group by the 6 pipeline steps.\n' +
  '2. Recommended reading order to follow the pipeline end-to-end.\n' +
  '3. The 3-5 implementation details worth emphasizing in the presentation / Q&A (concrete things visible in code that ground the paper, e.g. the exact sampling call, the permutation, the filter thresholds).\n' +
  '4. Any deviations between code and paper, or framework noise to ignore.\n' +
  'Write in Chinese. Base it strictly on the mappings below.\n\nPER-FILE MAPPINGS (JSON):\n' +
  JSON.stringify(results)

const synth = await agent(synthPrompt, { label: 'synthesize', phase: 'Synthesize' })

return { synth, results }
