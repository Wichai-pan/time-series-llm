# Code 组件占位

观察到的残留代码仓库现在集中放在：

```text
../legacy/old-project-files/puhti-time-series-llm/
```

它没有被复制到这里，因为该仓库有未核验的本地修改和 untracked 文件，当前不能作为可靠实现证据。现阶段只能把它当作之后要检查的 source。

reset 结束后需要选择一种策略：

- 继续使用 `legacy/old-project-files/puhti-time-series-llm/` 作为代码组件，并更新 `memory/component-index.yaml`；
- 重新创建一个干净的 `code/` repo；
- 在 `code-worktrees/` 下为已核验实验创建 worktree。
