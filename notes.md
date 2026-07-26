# Review PDF project relevance

> Read docs/run_selective_inference.md and the two pdf files in presentation/examples. Are there any information in the two pdf files in presentation/examples that I can use in this project.

<details><summary>6 previous messages</summary>

> I’m going to compare the project’s selective-inference workflow with both example PDFs, including equations, implementation details, diagnostics, and presentation ideas that transfer cleanly. I’m using the PDF skill because the source material is layout-sensitive and needs both extraction and visual verification.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Raw 'C:\Users\shida\.codex\plugins\cache\openai-primary-runtime\pdf\26.715.12143\skills\pdf\SKILL.md'`
> - Ran `Get-ChildItem -LiteralPath 'docs','presentation\examples' -Force | Select-Object FullName,Name,Length,LastWriteTime; Get-Content -Raw -LiteralPath 'docs\run_selective_inference.md'`
> - Ran `$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); pdfinfo 'presentation\examples\250728_shiraishi.pdf'; pdfinfo 'presentation\examples\250908_shiraishi.pdf'; pdftotext -layout -enc UTF-8 'presentation\examples\250728_shiraishi.pdf' 'tmp_250728.txt'; pdftotext -layout -enc UTF-8 'presentation\examples\250908_shiraishi.pdf' 'tmp_250908.txt'; Get-Content -Raw -Encoding UTF8 'tmp_250728.txt'; Get-Content -Raw -Encoding UTF8 'tmp_250908.txt'`
>
> </details>
>
> Poppler isn’t available in this Windows environment, so I’m following the skill’s fallback: extracting with the installed Python PDF stack and then rendering representative pages through an available PDF renderer if present. I’ll distinguish genuinely reusable material from content already captured in the current documentation.
>
> <details><summary>Ran 9 commands</summary>
>
> - Ran `$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); & '.\.venv\Scripts\python.exe' -c "import importlib.util; print('pypdf',bool(importlib.util.find_spec('pypdf'))); print('pdfplumber',bool(importlib.util.find_spec('pdfplumber'))); print('fitz',bool(importlib.util.find_spec('fitz'))); print('pymupdf',bool(importlib.util.find_spec('pymupdf')))"; Get-ChildItem -Path 'C:\Users\shida\.codex' -Recurse -ErrorAction SilentlyContinue -Include 'pdftotext.exe','pdftoppm.exe','mutool.exe' | Select-Object -First 20 -ExpandProperty FullName`
> - Ran `Get-ChildItem -Force; Get-ChildItem -Path '.venv','.venv-new' -Force -ErrorAction SilentlyContinue | Select-Object FullName,Name,Length; Get-Command python,py -ErrorAction SilentlyContinue | Select-Object Name,Source; py -0p`
> - Ran `Get-Content '.venv\pyvenv.cfg'; Test-Path 'C:\Users\shida\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe'; & 'C:\Users\shida\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe' -c "import sys,importlib.util; print(sys.version); print('pypdf',bool(importlib.util.find_spec('pypdf'))); print('pdfplumber',bool(importlib.util.find_spec('pdfplumber'))); print('fitz',bool(importlib.util.find_spec('fitz')))"`
> - Ran `where.exe /R 'C:\Program Files' pdftotext.exe 2>$null; where.exe /R 'C:\Program Files' mutool.exe 2>$null; where.exe /R 'C:\Program Files' pdftoppm.exe 2>$null; where.exe /R 'C:\Program Files (x86)' pdftotext.exe 2>$null; where.exe /R 'C:\Users\shida\AppData\Local' pdftotext.exe 2>$null; where.exe /R 'C:\Users\shida\AppData\Local' python.exe 2>$null`
> - Ran `$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); & 'C:\Program Files\Git\mingw64\bin\pdfinfo.exe' 'presentation\examples\250728_shiraishi.pdf'; & 'C:\Program Files\Git\mingw64\bin\pdfinfo.exe' 'presentation\examples\250908_shiraishi.pdf'; New-Item -ItemType Directory -Force -Path 'tmp\pdfs' | Out-Null; & 'C:\Program Files\Git\mingw64\bin\pdftotext.exe' -layout -enc UTF-8 'presentation\examples\250728_shiraishi.pdf' 'tmp\pdfs\250728.txt'; & 'C:\Program Files\Git\mingw64\bin\pdftotext.exe' -layout -enc UTF-8 'presentation\examples\250908_shiraishi.pdf' 'tmp\pdfs\250908.txt'; Get-Content -Raw -Encoding UTF8 'tmp\pdfs\250728.txt'; Get-Content -Raw -Encoding UTF8 'tmp\pdfs\250908.txt'`
> - Ran `Get-ChildItem -LiteralPath 'C:\Program Files\Git\mingw64\bin' -Filter 'pdf*.exe' | Select-Object -ExpandProperty FullName; Get-ChildItem -LiteralPath 'C:\Program Files\Git\mingw64\bin' -Filter 'pdftoppm.exe' | Select-Object -ExpandProperty FullName; Get-ChildItem -LiteralPath 'C:\Program Files\Git\mingw64\bin' -Filter 'pdftocairo.exe' | Select-Object -ExpandProperty FullName`
> - Ran `Get-Command magick,gswin64c,mutool,ffmpeg -ErrorAction SilentlyContinue | Select-Object Name,Source`
> - Ran `$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); rg --files presentation; rg -n "AIS|importance|ESS|balance|区間|FPR|SHAP|提案分布|sampling|mixture|両側" docs README.md src examples tests presentation -g '!presentation/examples/*.pdf'`
> - Ran ``$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); $d = Get-Content -Encoding UTF8 'docs\run_selective_inference.md'; $d[560..1060] -join "`n"; Get-Content -Encoding UTF8 'src\si_shap\inference.py'; Get-Content -Encoding UTF8 'src\si_shap\selection_regions.py'; Get-Content -Encoding UTF8 'presentation\presentation.tex'``
>
> </details>
>
> The substantive overlap is now clear: the July deck contains an earlier, more complex AIS design; the September deck is especially valuable as a failure analysis. I’m checking the plots/layout now, then I’ll give you a prioritized “use / adapt / don’t copy” assessment tied to the current code.
>
> <details><summary>Deleted 2 files, ran 2 commands, called a tool</summary>
>
> - Ran `Get-Content -Raw -LiteralPath 'C:\Users\shida\.codex\plugins\cache\openai-bundled\browser\26.707.91948\skills\control-in-app-browser\SKILL.md'`
>
> MCP tool call
>
> node_repl.js
>
> ```json
> {
>   "code": "if (globalThis.agent?.browsers == null) {\n  const { setupBrowserRuntime } = await import(\"C:/Users/shida/.codex/plugins/cache/openai-bundled/browser/26.707.91948/scripts/browser-client.mjs\");\n  await setupBrowserRuntime({ globals: globalThis });\n}\nif (globalThis.browser == null) {\n  globalThis.browser = await agent.browsers.getDefault();\n  nodeRepl.write(await browser.documentation());\n}",
>   "title": "Connect to PDF viewer"
> }
> ```
>
> ```text
> No browser is available
> ```
>
> - Ran `rg -n "^## (9|10|11|12|13|15|18)|^### 11\.3|^### 13\.1|^### 13\.2|^### 13\.3|^### 15\.1|^### 18\.3|^### 18\.7" docs/run_selective_inference.md; rg -n "DEFENSIVE_MIXTURE_WEIGHTS|def _adapt_proposal|def _sample_defensive_mixture|def _run_ais|denominator_ess|tail_ess|mc_se" src/si_shap/inference.py; rg -n "narrower than one grid cell|def find_selection_intervals|def compute_selection_regions" src/si_shap/selection_regions.py; rg -n "def plot_selection_regions|selected|proposal" src/si_shap/plotting.py`
>
> - Deleted `./tmp/pdfs/250728.txt` (+0 -333)
> - Deleted `./tmp/pdfs/250908.txt` (+0 -159)
>
> </details>
</details>

Yes. The PDFs contain useful material, mainly as AIS design guidance and failure analysis. Most mathematical foundations are already covered in [run_selective_inference.md](/C:/Users/shida/OneDrive/selective-inference/shap-selective-inference/docs/run_selective_inference.md:558), but the PDFs add practical warnings that should influence experiments and diagnostics.

| PDF material | Relevance to this project | Recommendation |
|---|---|---|
| Defensive mixture with minimum component weights (July, slides 9–10) | The project already implements a stronger version: 25% of the proposal is the target chi distribution, bounding raw importance weights by 4. | Keep the current implementation. Document that it addresses the support-loss problem described in the PDF. |
| Interval-aware TN and GMM proposals (July, slides 9–10) | Potentially useful when SHAP selection regions are disconnected or multimodal. The current adapted proposal is only one truncated normal. | Consider as an optional future proposal, especially for cases like the non-connected regions already found by this project. |
| Target/proposal plots with samples marked inside and outside the selection region (July, slides 9–17) | Very useful for diagnosing why adaptation succeeds or fails. Current plots show regions and densities but not sampled points or their weights. | Add a diagnostic plot showing pilot/final samples, selected status, tail status, and optionally normalized weight. |
| Fixed-proposal failure modes (September, slide 3) | Directly relevant: low ESS, isolated selected samples in the tail, and missed selection intervals. | Use these as named failure categories in documentation and experiment reports. |
| FPR remained 7.1% ± 1.6% despite apparently reasonable ESS and no exact 0/1 p-values (September, slide 4) | This is the most important lesson: ESS is necessary but does not establish selective-p-value validity. | Run large null-calibration experiments and inspect FPR and the full p-value distribution, not just `status`, ESS, and `mc_se`. |
| Narrow/disconnected interval omission (September, slide 6) | Directly relevant to the numerical region scanner, which already warns that intervals narrower than one grid cell may be missed. | Add multi-resolution or shifted-grid sensitivity checks when selection regions are used for validation or proposal construction. |
| Highly variable computation, including runs exceeding 96 hours (September, slide 6) | Relevant because every candidate response refits the forest and recomputes SHAP. | Continue using hard proposal budgets, explicit failure statuses, and failure-rate reporting. Also record wall-clock time per feature. |
| Discarding generations when their ESS is low (September, slides 5–6) | Statistically delicate: the decision to retain samples depends on observed samples and can introduce hard-to-characterize behavior. | Do not copy this. The current design—use pilot samples only for adaptation, then draw fresh samples from one fixed final mixture—is cleaner. |
| One-sided versus two-sided testing discussion (July, slides 3–6) | Only partially relevant. Those slides concern signed linear statistics, whereas this project uses the nonnegative norm statistic \(T=\|Q^\top y\|/\sigma\), an omnibus spline-effect test. | Useful background, but it should not replace the current chi upper-tail test. |

The current implementation already incorporates several lessons from the PDFs:

- A defensive target component in [inference.py](/C:/Users/shida/OneDrive/selective-inference/shap-selective-inference/src/si_shap/inference.py:12).
- Fresh sampling from a fixed final mixture after adaptation, avoiding problematic reuse of adaptive generations.
- Separate denominator and tail ESS checks in [inference.py](/C:/Users/shida/OneDrive/selective-inference/shap-selective-inference/src/si_shap/inference.py:208).
- Monte Carlo standard error and explicit non-convergence statuses.
- Selection-region visualization and an explicit warning about missed narrow intervals in [selection_regions.py](/C:/Users/shida/OneDrive/selective-inference/shap-selective-inference/src/si_shap/selection_regions.py:63).

One technical caution: the July deck describes all adaptive-generation samples as IID from the average proposal under a balance heuristic. They are not literally IID from that mixture; balance-heuristic estimators can still be constructed, but their justification must account for sampling from different proposals. The project’s current approach avoids this issue by discarding pilot samples and using IID draws from the final fixed mixture.

My priority recommendations are therefore:

1. Add a null-calibration/stress-testing workflow over several \(n,d,k\), forest settings, and AIS budgets.
2. Report FPR together with confidence intervals, p-value histograms, failure rate, ESS fractions, and `mc_se`.
3. Add sample-level proposal diagnostics to the existing selection-region plots.
4. Investigate an interval-aware multimodal proposal only if the current single adapted component produces poor tail ESS or frequent failures.
5. Treat numerical selection-region scans as diagnostics unless sensitivity to grid resolution has been checked.

So, yes—the PDFs are useful, but mainly for validating and diagnosing the current AIS implementation rather than supplying equations that should be copied directly.


---

To-Do

* Consider different ways for defining feature importance
* Consider different ways to deal with ties in feature importance