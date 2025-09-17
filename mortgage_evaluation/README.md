# mortgage_evaluation

该仓库提供了一个纯 Python 实现的公积金贷款抵押估值预测与风控流程。

## 快速开始

```bash
cd mortgage_evaluation
python -m mortgage.train_risk_model
```

执行完命令后会完成以下工作：

1. 加载 `data/sample_gjj_mortgage.csv` 中的样本数据（若文件不存在会自动生成）。
2. 基于线性回归训练抵押估值预测模型并计算 MAE、RMSE、R² 等指标。
3. 根据预测的贷款价值比（LTV）输出 `model_artifacts/` 目录下的结果文件：
   - `risk_model.json`：模型参数（截距、各特征系数、标准化统计量）。
   - `metrics.json`：测试集指标。
   - `risk_report.csv`：逐笔贷款的风险打分结果。
   - `risk_summary.csv`：风险分层的聚合统计。

## 数据说明

仓库附带的 `sample_gjj_mortgage.csv` 为合成样本，包含 800 条贷款记录与以下关键字段：

- 借款基本信息：贷款金额、利率、期限、收入、公积金缴存等；
- 抵押物估值：初始评估价值、模型预测的抵押价值；
- 风险标签：基于预测 LTV 自动划分的低/中/高风险层级。

用户可以替换为真实业务数据，保持列名一致即可复用风控流程。