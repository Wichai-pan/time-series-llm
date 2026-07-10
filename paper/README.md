# Report (LaTeX)

课程报告工程,已初始化好 —— 直接 `git pull` 后往 `main.tex` 里各写各节即可。

## 怎么编译

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# 或用 latexmk -pdf main.tex；也可整目录丢进 Overleaf
```

已在本机 `pdflatex` 验证过能编译(骨架 1 页)。

## 模板 = NeurIPS 2026 style,preprint 模式

```latex
\usepackage[preprint]{neurips_2026}
```
- **preprint** = 非匿名(显示作者)、页脚只印 "Preprint."。**不用 `final`**:2026 的 final 需要指定 track(如 `[final,main]`)且页脚会印 "NeurIPS 2026 会议",课程报告不合适。

## 目录

- `main.tex` —— 主文件,已按下面 6 个模块排好骨架(每节留了 `% TODO`)
- `neurips_2026.sty` —— 模板样式(勿改)
- `neurips_2026.tex` / `checklist.tex` —— 官方示例 / 检查表(参考用)
- `figures/` —— 放图
- `refs.bib` —— 参考文献(起始有 PAMAP2/TSGBench,待补 + 核对)
- 编译产物(*.aux/*.log/*.pdf 等)已 `.gitignore`,不提交

## 必须包含的 6 个模块(导师 6-22 定,格式不限但内容必须齐)

1. Introduction(research question)
2. Literature review
3. Methods
4. **Experiments**(试过的模型 / 调过的参数 / 各结果)
5. **Result analysis**(什么 setup 好/为什么,什么不好/为什么)
6. Conclusion / Future work

## 待办

- 补作者名/顺序/邮箱(`main.tex` 里现在是占位)
- 各自填 6 节内容

时间线:~7/10 给导师 draft → 7/19 final deadline。
