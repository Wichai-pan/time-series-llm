# Report (LaTeX)

课程报告工程目录。模板由你手动下载后放到这里。

## 下载哪个模板

**NeurIPS 2024 style(单栏)** —— 官方 style files:
- 页面:https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
- 下载后是个 zip,里面有 `neurips_2024.sty`、`neurips_2024.tex`(示例正文)、`neurips_2024.bib`。
- **把这些文件解压到本目录**(`paper/`),`main.tex` 就用 `neurips_2024.tex` 改名/改写。

## 关键:用 final 模式(非匿名、显示作者)

```latex
\usepackage[final]{neurips_2024}   % 课程报告要署名,别用默认的匿名/preprint
```

## 目录约定

- `main.tex` —— 主文件(从 neurips_2024.tex 改)
- `figures/` —— 放图(已建好)
- `refs.bib` —— 参考文献

## 必须包含的 6 个模块(导师 6-22 定,格式不限但内容必须齐)

1. Introduction(research question)
2. Literature review
3. Methods
4. **Experiments**(试过的模型 / 调过的参数 / 各结果)
5. **Result analysis**(什么 setup 好/为什么,什么不好/为什么)
6. Conclusion / Future work

时间线:~7/10 给导师 draft → 7/19 final deadline。
