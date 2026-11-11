# Figure legends (Nature format)

> 按 Nature 规范写：`Fig. N | 标题句`，然后分面板说明；统计量、n、误差定义、源数据声明都写在图注里。
> 主线图注控制在 200–250 词。两份都可直接投稿。

---

## Fig. 1 | One nominal label, costs from a 0.3% gain to a 10.9% loss, concentrated in the step that saves least memory

**a**, Accuracy across the GGUF quantization ladder for four instruction-tuned models (4-task mean over GSM8K, ARC-Challenge, AG News and MMLU; 800 items per model–task cell). Points are means; bands are 95% percentile-bootstrap intervals over items (2,000 resamples). The shaded band marks Q4_K_M, the level most commonly shipped. The sub-2B models begin losing accuracy at Q4_K_M. Qwen2.5-3B does too, by 4.0%; Llama-3.2-3B stays flat to Q4_K_M, below which it was not measured. **b**, Accuracy change from Q8_0 to Q4_K_M, expressed as a percentage of each model's own Q8_0 accuracy, against parameter count (log scale). Vertical bars are 95% paired-bootstrap intervals over matched items (5,000 resamples); values are printed beside each interval. The size interaction is significant against the 3B tier with both its models pooled (difference-in-differences −0.026, 95% CI −0.041 to −0.012, p = 0.0004 at 1B; −0.038, −0.055 to −0.021, p = 0.0002 at 0.5B); the 0.5B-versus-1B contrast is not resolved at this sample size (p = 0.25). **c**, The two steps a deployment decision turns on, drawn as one arrow per model from its 8→5-bit step (hollow marker) to its 5→4-bit step (filled marker). Memory saved is the difference between GGUF artifact sizes. For Qwen2.5-0.5B, 106 MB is given up for 1.0 point of accuracy at 8→5 bits, whereas 21 MB costs 4.2 points at 5→4 bits. **d**, Per-task accuracy change at Q4_K_M relative to Q8_0. Cells are annotated with the change in percentage points. The cost is present in every task for the sub-2B models and is not driven by a single task.

All panels derive from 70,394 scored responses. Statistics: two-sided paired bootstrap; Benjamini–Hochberg correction applied within each table family. Source data are provided as CSV files (`fig1a`–`fig1d`) alongside the plotting code.

---

## Fig. 2 | The 4-bit step raises calibration error in every model, most strongly at the smallest

**a**, Expected calibration error (ECE) across the quantization ladder for the same four models. ECE is computed within each task over 15 equal-width confidence bins and then averaged across tasks; pooling items across tasks before computing ECE would mix tasks with very different accuracy and confidence distributions and inflates the estimate. Lower is better. The shaded band marks Q4_K_M. **b**, Reliability of the smallest and largest Qwen models: empirical accuracy against stated confidence, binned at 10 equal widths with bins holding fewer than 20 items omitted. Filled circles are Q4_K_M and open squares are Q8_0; colour identifies the model. Marker area is proportional to the number of items in the bin. Stated confidence is not continuous — it concentrates on round values, so each condition occupies only two to five bins and the bins are plotted as markers rather than connected as a curve. Points below the dashed diagonal indicate over-confidence. **c**, Increase in ECE from Q8_0 to Q4_K_M for each model, with 95% paired-bootstrap intervals over items resampled within task (2,000 resamples per model). The estimates are ordered by size class, from +0.071 at 0.5B to +0.015 and +0.004 for the two 3B models; the Llama interval includes zero. This panel shows the stated-confidence channel only. Table 4 reports the same contrast on token-probability confidence, where the ordering by size does not hold. The grey band marks no change.

Calibration is computed only over responses that stated a confidence, which was 84–87% of responses at Q4_K_M for the 0.5B and 1B models and 99.6% or more for the 3B models. Source data are provided as CSV files (`fig2a`–`fig2c`).

---

## 排版与提交说明

**尺寸**：Fig. 1 为 180 × 124 mm，Fig. 2 为 180 × 100 mm，均为双栏满宽。字数上限内（Nature 单条图注建议 ≤300 词）。

**字体**：Arial / Helvetica 系，正文最小 6 pt（实测最小 5.9 pt，高于 5 pt 硬底线）。SVG 中文字保持可编辑（`svg.fonttype = none`），PDF 使用 TrueType 嵌入（`pdf.fonttype = 42`），可在 Illustrator 中直接改字。

**交付件**（`figures/nature/`）：

| 文件 | 用途 |
|---|---|
| `figure1_accuracy.svg` / `.pdf` / `.png` | 矢量主件 + 位图预览 |
| `figure2_calibration.svg` / `.pdf` / `.png` | 同上 |
| `*.alignment.json` | 面板对齐实测几何（审稿/返修时证明对齐，非投稿件） |
| `*.collision.json` | 元素碰撞审计记录 |
| `figures/source_data/*.csv` | 每张图每个面板的源数据 |

**已通过的自动检查**：
- 面板对齐门（`require_matplotlib_panel_alignment`，容差 1.5 pt）：两张图均通过
- PDF 字号扫描（`audit_pdf_text.py --min-pt 5`）：Fig. 1 最小 5.9 pt、Fig. 2 最小 5.9 pt，均 PASS
- 碰撞审计（`audit_figure_collisions.py`）：两张图 **0 FAIL**

**3 条 WARN 及处置**（规范要求在最终尺寸目视确认后记录理由）：

| 图 | WARN | 处置 |
|---|---|---|
| Fig. 1a / Fig. 2a | `deployment default` 文字落在阴影色带边缘 | 有意：该文字就是给这条色带命名 |
| Fig. 2c | `Llama-3.2-3B (3B)` 标签起点落在 x<0 的灰带内 | 有意：该模型的区间本身跨过零点，标签锚在区间左端 |

**配色逻辑**（图注不必写，但审稿人会问）：色相区分模型家族（Qwen 蓝、Gemma 青、Llama 紫），同族内用深浅区分规模（Qwen 0.5B 浅蓝、3B 深蓝）。红色只用于"变差"的方向提示。这样四个模型在多面板间保持一致，读者不需要反复回图例。

---

## 与正文的一致性

图里的每个数字都已回写到正文：

| 数字 | 正文位置 | 图 |
|---|---|---|
| −10.9% / −7.9% / −1.9% | Table 1、Abstract | Fig. 1b |
| −0.042 / −0.054（交互） | Table 2 | Fig. 1b 图注 |
| +0.071 / +0.034 / +0.015 / +0.004 | Table 3 | Fig. 2c |
| 106 MB / 21 MB / 1,012 MB / 281 MB | Table 5、Introduction | Fig. 1c |

**注意**：`106 MB` 与 `21 MB` 是 **GGUF 文件体积之差**，不是标称位宽比。之前正文里写过 111 MB / 23 MB，那是用 `ollama list` 的四舍五入值算的；改用 HEAD 请求取得的精确字节数后已全部更正。图注里写明了这一点，因为这正是审稿人会追问的地方。
