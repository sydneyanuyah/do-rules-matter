# Corrected HEF backbone protocol

Primary `HEF with X` means the text backbone X and the structured-evidence fusion
head are optimized jointly. Gradients flow from the classification loss through
both components. Test is scored once after learning rate, epoch, and threshold
are selected on validation.

Historical scikit-learn HEF results use a frozen scalar similarity and are retained
only as `HEF with frozen X` ablations. They are not renamed or substituted for the
jointly fine-tuned model.

The joint neural HEF head is used because GBDT is non-differentiable and cannot
fine-tune a Transformer backbone end-to-end.
