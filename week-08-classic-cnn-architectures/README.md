# Week 8: Classic CNN Architectures + Regularization (Days 49–55)

Continuing the "recreate a real historical architecture, honestly" thread from Days 47–48. Currently in progress — only Day 49 is built so far; Days 50–55 (dropout from scratch, L1/L2 on CNNs, VGG-style stacked convs, residual connections, batch norm in CNNs, and data-loading pipelines) are still ahead. See [`full_syllabus_days_1_365.md`](../full_syllabus_days_1_365.md) for what's planned.

## Day 49 — AlexNet-style deeper CNN
[`day49_alexnet_style_cnn.py`](day49_alexnet_style_cnn.py) — rebuilds AlexNet's (2012) defining ideas at this series' scale: ReLU throughout, a third stacked conv+pool block, two big dropout-regularized FC layers, on larger 40x40 images, trained with real mini-batches for the first time since Day 45's mechanism. New from-scratch code: a standalone inverted-dropout layer, verified by both a pooled statistical check and a full gradient check.

Honest result: on a fixed 40-epoch budget, this series' own much smaller 2-block baseline beats AlexNetStyle outright (test_acc=0.613 vs. 0.363) — two stacked dropout layers cut how much signal reaches the FC weights per step, so the deeper, dropout-heavy network needs more training steps to catch up. A follow-up ablation confirms dropout does narrow the train/test gap as intended, just not within this epoch budget.

Study guide: [`day49_alexnet_style_cnn_complete_guide.pdf`](day49_alexnet_style_cnn_complete_guide.pdf) · Syntax walkthrough: [`day49_syntax_line_by_line.pdf`](day49_syntax_line_by_line.pdf) · [Run output](day49_run_output.txt)

![AlexNetStyle vs. baseline loss](alexnet_vs_baseline_loss.png)
![Dropout ablation](alexnet_dropout_ablation.png)

---
[← Back to main README](../README.md)
